# app.py — Expert Card Multi-Agents (Director + Web Search + Vision + Finalizer)
# Robust gegen nicht-wohlgeformtes JSON: sichere Parser + Fallbacks.
# - Natürliches Gespräch (Director, GPT-5), ohne redundante Bestätigungen
# - Aktionen (search_cover) werden autonom ausgelöst
# - Hintergrund-Pipeline: Websuche → Vision-Validierung → Finalizer
# - UI: Bild links, finaler Absatz rechts (pro Topic)

import os
import re
import json
import uuid
import threading
import tempfile
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from openai import OpenAI
import requests

# =======================
# CONFIG
# =======================
APP_TITLE = "Expert Card – Mini Agents (Vision)"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Modelle (Primär + Fallbacks)
CHAT_MODEL_PRIMARY = os.getenv("DTBR_CHAT_MODEL", "gpt-5")
CHAT_MODEL_FALLBACKS = ["gpt-4o", "gpt-4o-mini"]

SEARCH_MODEL_PRIMARY = os.getenv("DTBR_SEARCH_MODEL", "gpt-5")  # Responses + web_search
SEARCH_MODEL_FALLBACKS = ["gpt-4.1", "gpt-4o"]

# Downloads
ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/webp"}
HTTP_TIMEOUT = 20

# UI Debug Toggle (per Checkbox ein-/ausschaltbar)
if "SHOW_DEBUG" not in st.session_state:
    st.session_state.SHOW_DEBUG = False

# =======================
# SESSION STATE
# =======================
def ensure_session():
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_director_prompt()},
            {"role": "assistant", "content":
                "Hi! I'll help you craft a tiny expert card. "
                "First, which book has helped you professionally? "
                "Please give the title (author optional)."}
        ]
    if "profile" not in st.session_state:
        st.session_state.profile = {"topics": []}  # siehe Topic-Objekte unten
    if "tasks" not in st.session_state:
        st.session_state.tasks: Dict[str, Dict[str, Any]] = {}  # id -> {status, kind, ...}
    if "results" not in st.session_state:
        st.session_state.results: List[Dict[str, Any]] = []      # fertige Kacheln (Bild + Text)
    if "last_director_json" not in st.session_state:
        st.session_state.last_director_json = None
    if "last_raw_director_text" not in st.session_state:
        st.session_state.last_raw_director_text = ""

def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set (Streamlit → Settings → Secrets).")
    return OpenAI(api_key=OPENAI_API_KEY)

# =======================
# PROMPTS
# =======================
def system_director_prompt() -> str:
    return (
        "You are the Interview Director for a tiny 'Expert Card'.\n"
        "LANGUAGE:\n"
        "- Default to English; mirror the user's language if clearly not English.\n"
        "GOALS:\n"
        "- Collect for topics like book/podcast/role_model/tool:\n"
        "  • book: title (required), optional author_guess; why (1–2 sentences)\n"
        "  • podcast: title; why (1–2 sentences)\n"
        "  • role_model: name; why (1–2 sentences)\n"
        "  • tool: title; why (1–2 sentences)\n"
        "BEHAVIOR:\n"
        "- Keep conversation warm and natural. Ask targeted follow-ups that reveal the user's thinking.\n"
        "- NEVER ask redundant confirmations if information is already clear.\n"
        "- If the title/name is sufficiently unambiguous (or author provided), you MAY trigger a cover/portrait search immediately.\n"
        "- If multiple likely authors exist, ask ONE brief disambiguation question.\n"
        "- After 1–2 meaningful follow-ups per topic, move on to another topic (e.g., podcast → role model → tool).\n"
        "CONTROL OUTPUT (STRICT JSON):\n"
        "{\n"
        "  assistant_visible: string (short next message in user's language),\n"
        "  facts_partial: {\n"
        "    book?: {title?, author_guess?, why?},\n"
        "    podcast?: {title?, why?},\n"
        "    role_model?: {name?, why?},\n"
        "    tool?: {title?, why?},\n"
        "    extras?: object\n"
        "  },\n"
        "  actions: [\n"
        "    { type:'search_cover', kind:'book|podcast|person|tool', title?:string, author?:string, name?:string }\n"
        "  ]\n"
        "}\n"
        "Output ONLY the JSON object. No prose, no backticks."
    )

def director_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "assistant_visible": {"type": "string"},
            "facts_partial": {
                "type": "object",
                "properties": {
                    "book": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "author_guess": {"type": "string"},
                            "why": {"type": "string"}
                        }
                    },
                    "podcast": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "why": {"type": "string"}
                        }
                    },
                    "role_model": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "why": {"type": "string"}
                        }
                    },
                    "tool": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "why": {"type": "string"}
                        }
                    },
                    "extras": {"type": "object"}
                }
            },
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["search_cover"]},
                        "kind": {"type": "string", "enum": ["book", "podcast", "person", "tool"]},
                        "title": {"type": "string"},
                        "author": {"type": "string"},
                        "name": {"type": "string"}
                    },
                    "required": ["type", "kind"]
                }
            },
            "notes": {"type": "string"}
        },
        "required": ["assistant_visible", "facts_partial", "actions"]
    }

def websearch_system_prompt(kind: str) -> str:
    base = (
        "You are a web image search assistant.\n"
        "Find up to three HIGH-QUALITY image candidates for the requested item.\n"
        "Return ONLY JSON with: candidates: [ {image_url, page_url, source, title, authors[]} ].\n"
        "Prefer trusted sources: publisher, Amazon, Goodreads, Open Library, Wikipedia/Commons, or the official site/channel.\n"
        "No extra text; JSON only."
    )
    if kind == "book":
        return base + "\nEnsure the images are book covers (portrait aspect, visible title/author typical of covers)."
    if kind == "podcast":
        return base + "\nLook for podcast/youtube channel artwork (logo/title)."
    if kind == "person":
        return base + "\nLook for portraits (press photos, Wikipedia portrait)."
    if kind == "tool":
        return base + "\nLook for product/app logos or official product shots."
    return base

def validator_system_prompt(kind: str) -> str:
    return (
        "You are a VISION validator.\n"
        "Inputs:\n"
        "- expected (title/author/name) strings,\n"
        "- up to 3 candidate images (as image inputs) with metadata (page_url, source, scraped title/authors if available).\n"
        "Task:\n"
        f"- Inspect each image VISUALLY to judge whether it is a plausible {kind} image.\n"
        "- Verify if visible title/author/name align with expected strings (tolerate minor punctuation/case differences).\n"
        "- Prefer trustworthy sources (publisher, Amazon, Goodreads, Open Library, official site) if images look similar.\n"
        "Output STRICT JSON: {best_index: 0|1|2|-1, reason: string, confidence: number}. No prose."
    )

def finalizer_system_prompt() -> str:
    return (
        "You are a finalizing writer. Create a professional, warm, uplifting 3–5 sentence paragraph that presents the user positively without flattery. "
        "Use the interview facts (why the item matters to them) and the verified media (title/author/name). Be specific, concise, and human."
    )

# =======================
# JSON PARSER (robust)
# =======================
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

def strip_code_fences(s: str) -> str:
    return CODE_FENCE_RE.sub("", s).strip()

def extract_balanced_json(s: str) -> Optional[str]:
    """
    Find the first balanced {...} block and return it. Handles nested braces, quotes, and escapes.
    """
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None

def parse_json_strict_or_best_effort(text: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Try to parse JSON. If it fails, try to strip code fences and extract a balanced JSON block.
    Returns (data, raw_used_text) — data is dict or None; raw_used_text shows what we attempted.
    """
    raw = text or ""
    try:
        return json.loads(raw), raw
    except Exception:
        pass
    cleaned = strip_code_fences(raw)
    try:
        return json.loads(cleaned), cleaned
    except Exception:
        pass
    maybe = extract_balanced_json(cleaned)
    if maybe:
        try:
            return json.loads(maybe), maybe
        except Exception:
            return None, maybe
    return None, cleaned

# =======================
# OPENAI WRAPPER
# =======================
def responses_create(client: OpenAI, **kwargs):
    # leichte „Deterministik“
    kwargs.setdefault("temperature", 0.2)
    try:
        return client.responses.create(**kwargs)
    except TypeError:
        cleaned = dict(kwargs)
        cleaned.pop("response_format", None)
        cleaned.pop("tools", None)
        return client.responses.create(**cleaned)

# =======================
# PROFILE / TOPICS
# =======================
def ensure_topic(profile: Dict[str, Any], kind: str, key_value: str) -> int:
    topics = profile["topics"]
    key_field = "name" if kind == "person" else "title"
    for i, t in enumerate(topics):
        if t.get("kind") == kind and (t.get("facts", {}).get(key_field, "") or "").lower() == (key_value or "").lower():
            return i
    topic = {
        "kind": kind,
        "facts": {key_field: key_value},
        "status": "collecting",
        "media": {}
    }
    topics.append(topic)
    return len(topics) - 1

def merge_facts(profile: Dict[str, Any], partial: Dict[str, Any]):
    if not partial:
        return
    # BOOK
    if "book" in partial and isinstance(partial["book"], dict):
        title = partial["book"].get("title", "") or profile_key(profile, "book")
        idx = ensure_topic(profile, "book", title)
        profile["topics"][idx]["facts"].update({k: v for k, v in partial["book"].items() if isinstance(v, str) and v.strip()})
    # PODCAST
    if "podcast" in partial and isinstance(partial["podcast"], dict):
        title = partial["podcast"].get("title", "") or profile_key(profile, "podcast")
        idx = ensure_topic(profile, "podcast", title)
        profile["topics"][idx]["facts"].update({k: v for k, v in partial["podcast"].items() if isinstance(v, str) and v.strip()})
    # ROLE MODEL
    if "role_model" in partial and isinstance(partial["role_model"], dict):
        name = partial["role_model"].get("name", "") or profile_key(profile, "person")
        idx = ensure_topic(profile, "person", name)
        profile["topics"][idx]["facts"].update({k: v for k, v in partial["role_model"].items() if isinstance(v, str) and v.strip()})
    # TOOL
    if "tool" in partial and isinstance(partial["tool"], dict):
        title = partial["tool"].get("title", "") or profile_key(profile, "tool")
        idx = ensure_topic(profile, "tool", title)
        profile["topics"][idx]["facts"].update({k: v for k, v in partial["tool"].items() if isinstance(v, str) and v.strip()})

def profile_key(profile: Dict[str, Any], kind: str) -> str:
    for t in profile["topics"]:
        if t["kind"] == kind:
            key_field = "name" if kind == "person" else "title"
            return t.get("facts", {}).get(key_field, "")
    return ""

# =======================
# AGENT CALLS
# =======================
def call_director(client: OpenAI, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema = director_schema()
    payload = {
        "model": CHAT_MODEL_PRIMARY,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "director_control", "schema": schema, "strict": True}},
        "input": [
            {"role": "system", "content": system_director_prompt()},
            *history[2:]  # initial system + first assistant already in history
        ]
    }
    last_err = None
    for m in [CHAT_MODEL_PRIMARY, *CHAT_MODEL_FALLBACKS]:
        payload["model"] = m
        try:
            r = responses_create(client, **payload)
            # Raw-Text extrahieren
            text = getattr(r, "output_text", "")
            if not text and getattr(r, "output", None):
                blk = r.output[0]
                if blk and blk.get("content") and blk["content"][0].get("text"):
                    text = blk["content"][0]["text"]
            st.session_state.last_raw_director_text = text or ""
            data, used = parse_json_strict_or_best_effort(text or "")
            if data is not None:
                return data
            # Fallback: sichere Minimalstruktur
            return {
                "assistant_visible": "Got it! What makes it stand out for you? A sentence or two is enough.",
                "facts_partial": {},
                "actions": []
            }
        except Exception as e:
            last_err = e
            continue
    # Wenn gar nichts klappt, immer noch eine sichere Antwort zurückgeben
    if last_err:
        st.session_state.last_raw_director_text = f"(error: {last_err})"
    return {
        "assistant_visible": "Thanks! Tell me briefly why this matters to you.",
        "facts_partial": {},
        "actions": []
    }

def call_websearch(client: OpenAI, kind: str, title_or_name: str, author: str = "") -> Dict[str, Any]:
    payload = {
        "model": SEARCH_MODEL_PRIMARY,
        "tools": [{"type": "web_search"}],
        "response_format": {"type": "json_object"},
        "input": [
            {"role": "system", "content": websearch_system_prompt(kind)},
            {"role": "user", "content": json.dumps({
                "kind": kind,
                "title_or_name": title_or_name,
                "author": author
            }, separators=(",", ":"))}
        ]
    }
    last_err = None
    for m in [SEARCH_MODEL_PRIMARY, *SEARCH_MODEL_FALLBACKS]:
        payload["model"] = m
        try:
            r = responses_create(client, **payload)
            text = getattr(r, "output_text", "") or (r.output[0]["content"][0]["text"] if getattr(r, "output", None) else "")
            data, _ = parse_json_strict_or_best_effort(text or "")
            cands = (data or {}).get("candidates") or []
            return {"candidates": cands[:3]}
        except Exception as e:
            last_err = e
            continue
    # Fallback
    return {"candidates": []}

def call_vision_validator(client: OpenAI, kind: str,
                          expected_title: str, expected_author_or_name: str,
                          candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Text + bis zu 3 Bilder
    content_parts: List[Dict[str, Any]] = [
        {"type": "input_text",
         "text": json.dumps({
             "kind": kind,
             "expected_title": expected_title,
             "expected_author_or_name": expected_author_or_name,
             "candidates_meta": [
                 {
                     "page_url": c.get("page_url", ""),
                     "source": c.get("source", ""),
                     "title": c.get("title", ""),
                     "authors": c.get("authors", []),
                     "image_url": c.get("image_url", "")
                 } for c in candidates[:3]
             ]
         }, separators=(",", ":"))}
    ]
    for c in candidates[:3]:
        if c.get("image_url"):
            content_parts.append({"type": "input_image", "image_url": c["image_url"]})

    payload = {
        "model": CHAT_MODEL_PRIMARY,  # Vision-fähig
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "cover_validation",
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "best_index": {"type": "integer"},
                                                    "reason": {"type": "string"},
                                                    "confidence": {"type": "number"}
                                                },
                                                "required": ["best_index", "reason", "confidence"]
                                            },
                                            "strict": True}},
        "input": [
            {"role": "system", "content": validator_system_prompt(kind)},
            {"role": "user", "content": content_parts}
        ]
    }
    last_err = None
    for m in [CHAT_MODEL_PRIMARY, *CHAT_MODEL_FALLBACKS]:
        payload["model"] = m
        try:
            r = responses_create(client, **payload)
            text = getattr(r, "output_text", "") or (r.output[0]["content"][0]["text"] if getattr(r, "output", None) else "")
            data, _ = parse_json_strict_or_best_effort(text or "")
            if data is None:
                return {"best_index": -1, "reason": "Parse failed", "confidence": 0.0}
            return data
        except Exception as e:
            last_err = e
            continue
    return {"best_index": -1, "reason": "Validator failed", "confidence": 0.0}

def call_finalizer(client: OpenAI, facts: Dict[str, Any], chosen_meta: Dict[str, Any]) -> str:
    payload = {
        "model": CHAT_MODEL_PRIMARY,
        "temperature": 0.6,  # finaler Text darf etwas wärmer sein
        "input": [
            {"role": "system", "content": finalizer_system_prompt()},
            {"role": "user", "content": json.dumps({
                "facts": facts,
                "verified_media": chosen_meta
            }, separators=(",", ":"))}
        ]
    }
    last_err = None
    for m in [CHAT_MODEL_PRIMARY, *CHAT_MODEL_FALLBACKS]:
        payload["model"] = m
        try:
            r = responses_create(client, **payload)
            txt = getattr(r, "output_text", "") or (r.output[0]["content"][0]["text"] if getattr(r, "output", None) else "")
            return (txt or "").strip()
        except Exception as e:
            last_err = e
            continue
    return "A concise, positive summary will appear here once the network recovers."

# =======================
# DOWNLOADER (nur für Anzeige)
# =======================
def safe_download_image(url: str) -> Optional[str]:
    try:
        with requests.get(url, timeout=HTTP_TIMEOUT, stream=True, headers={"User-Agent": "Mozilla/5.0"}) as r:
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if ctype not in ALLOWED_IMAGE_MIME:
                return None
            suf = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(ctype, ".img")
            fd, path = tempfile.mkstemp(prefix="cover_", suffix=suf)
            with os.fdopen(fd, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return path
    except Exception:
        return None

# =======================
# BACKGROUND TASKS
# =======================
def start_cover_task(client: OpenAI, kind: str, title_or_name: str, author: str, topic_index: int, facts_snapshot: Dict[str, Any]):
    task_id = str(uuid.uuid4())
    st.session_state.tasks[task_id] = {"status": "running", "kind": "cover", "topic_index": topic_index, "error": "", "result": None}

    def worker():
        try:
            st.session_state.profile["topics"][topic_index]["status"] = "searching"

            # 1) Websuche
            found = call_websearch(client, kind, title_or_name, author)
            candidates = (found.get("candidates") or [])[:3]
            st.session_state.profile["topics"][topic_index].setdefault("media", {})["candidates"] = candidates

            # 2) Vision-Validierung
            st.session_state.profile["topics"][topic_index]["status"] = "validating"
            expected_title = title_or_name if kind != "person" else ""
            expected_author_or_name = author if kind == "book" else (title_or_name if kind == "person" else "")
            val = call_vision_validator(client, kind, expected_title, expected_author_or_name, candidates)
            best_i = val.get("best_index", -1)
            reason = val.get("reason", "")
            conf = float(val.get("confidence", 0.0))

            chosen = candidates[best_i] if 0 <= best_i < len(candidates) else {}
            st.session_state.profile["topics"][topic_index]["media"]["validator"] = val
            st.session_state.profile["topics"][topic_index]["media"]["chosen"] = chosen

            # 3) Finalizer-Text
            final_txt = call_finalizer(client, facts_snapshot, chosen)

            res = {
                "topic_index": topic_index,
                "kind": kind,
                "image": {
                    "image_url": chosen.get("image_url"),
                    "page_url": chosen.get("page_url"),
                    "source": chosen.get("source")
                },
                "text": {"final_paragraph": final_txt},
                "meta": {"validator_reason": reason, "confidence": conf}
            }
            st.session_state.results.append(res)

            st.session_state.profile["topics"][topic_index]["status"] = "ready"
            st.session_state.tasks[task_id] = {"status": "done", "kind": "cover", "topic_index": topic_index, "error": "", "result": res}
        except Exception as e:
            st.session_state.profile["topics"][topic_index]["status"] = "failed"
            st.session_state.tasks[task_id] = {"status": "error", "kind": "cover", "topic_index": topic_index, "error": str(e), "result": None}

    threading.Thread(target=worker, daemon=True).start()

# =======================
# STREAMLIT UI
# =======================
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="centered")
colA, colB = st.columns([1, 1])
with colA:
    st.title("Dein Kurz-Steckbrief")
    st.caption("Director decides • parallel web search + vision validate • final shows as two columns")
with colB:
    st.toggle("Debug", key="SHOW_DEBUG", help="Show raw JSON/text outputs for troubleshooting")

ensure_session()

# Verlauf (ohne initiales system)
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Client
client = None
agent_ready = True
try:
    client = get_client()
except Exception as e:
    agent_ready = False
    st.error(f"OpenAI key missing: {e}")

# Chat Input
user_text = st.chat_input("Type your answer…")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})

    if agent_ready:
        # Director entscheidet (liefert sichtbaren Satz + actions + facts_partial)
        director_json = call_director(client, st.session_state.messages)
        st.session_state.last_director_json = director_json

        # Merge Facts
        merge_facts(st.session_state.profile, director_json.get("facts_partial") or {})

        # Sichtbare Antwort
        vis = (director_json.get("assistant_visible") or "").strip()
        if vis:
            st.session_state.messages.append({"role": "assistant", "content": vis})

        # Aktionen ausführen (parallel)
        for act in (director_json.get("actions") or []):
            if act.get("type") == "search_cover":
                kind = (act.get("kind") or "").strip()
                if not kind:
                    continue
                if kind == "person":
                    name = (act.get("name") or profile_key(st.session_state.profile, "person")).strip()
                    if not name:
                        continue
                    idx = ensure_topic(st.session_state.profile, "person", name)
                    if st.session_state.profile["topics"][idx]["status"] in ("queued", "searching", "validating"):
                        continue
                    st.session_state.profile["topics"][idx]["status"] = "queued"
                    snapshot = json.loads(json.dumps(st.session_state.profile["topics"][idx]["facts"]))
                    start_cover_task(client, "person", name, "", idx, {"topic": snapshot})
                else:
                    title = (act.get("title") or profile_key(st.session_state.profile, kind)).strip()
                    author = (act.get("author") or "").strip()
                    if not title:
                        continue
                    idx = ensure_topic(st.session_state.profile, kind, title)
                    if st.session_state.profile["topics"][idx]["status"] in ("queued", "searching", "validating"):
                        continue
                    st.session_state.profile["topics"][idx]["status"] = "queued"
                    snapshot = json.loads(json.dumps(st.session_state.profile["topics"][idx]["facts"]))
                    start_cover_task(client, kind, title, author, idx, {"topic": snapshot})

    else:
        st.session_state.messages.append({"role": "assistant",
                                          "content": "(Demo mode — set OPENAI_API_KEY in Streamlit Secrets.)"})

    st.rerun()

# Laufende Tasks
running = [t for t in st.session_state.tasks.values() if t["status"] == "running"]
if running:
    st.info("🔎 Background research running… you can keep chatting.")

# Ergebnisse
if st.session_state.results:
    st.subheader("Results")
    for res in st.session_state.results[::-1]:
        c1, c2 = st.columns([1, 1.25])
        with c1:
            img_url = res.get("image", {}).get("image_url")
            if img_url:
                st.image(img_url, caption=res.get("image", {}).get("source", ""), use_container_width=True)
            else:
                st.write("No image available.")
        with c2:
            kind = res.get("kind", "")
            # einfache Überschrift
            header = kind.capitalize() if kind else "Item"
            st.markdown(f"**{header}**")
            st.write(res.get("text", {}).get("final_paragraph", ""))
            page_url = res.get("image", {}).get("page_url")
            if page_url:
                st.markdown(f"[Source]({page_url})")
            if st.session_state.SHOW_DEBUG:
                with st.expander("Debug (result json)"):
                    st.code(json.dumps(res, indent=2))

# Controls
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🔄 Restart interview"):
        for k in ["messages", "profile", "tasks", "results", "last_director_json", "last_raw_director_text"]:
            if k in st.session_state:
                del st.session_state[k]
        ensure_session()
        st.rerun()
with c2:
    if st.session_state.SHOW_DEBUG and st.session_state.get("last_director_json") is not None:
        with st.expander("Last director JSON (parsed)"):
            st.code(json.dumps(st.session_state.last_director_json, indent=2))
with c3:
    if st.session_state.SHOW_DEBUG and st.session_state.get("last_raw_director_text"):
        with st.expander("Last director RAW text"):
            st.code(st.session_state.last_raw_director_text)
