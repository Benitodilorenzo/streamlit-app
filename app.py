# app.py — Expert Card Multi-Agents (Director + Web Search + Vision + Finalizer)
# Fixes:
# - Keine st.session_state-Zugriffe in Threads (nutzt BG_TASKS/BG_RESULTS + Harvest im Main-Thread)
# - Topic-Followups nur bei tatsächlichem Fakten-Delta des aktuellen Topics
# - Director optionales topic_focus; Fallback-Antwort zählt nicht als Follow-up
# - Actions dedupliziert via inflight_keys
# - Refresh-Button für Ergebnisse

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

# Modelle
CHAT_MODEL_PRIMARY = os.getenv("DTBR_CHAT_MODEL", "gpt-5")
CHAT_MODEL_FALLBACKS = ["gpt-4o", "gpt-4o-mini"]
SEARCH_MODEL_PRIMARY = os.getenv("DTBR_SEARCH_MODEL", "gpt-5")  # Responses + web_search
SEARCH_MODEL_FALLBACKS = ["gpt-4.1", "gpt-4o"]

ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/webp"}
HTTP_TIMEOUT = 20

TOPIC_ORDER = ["book", "podcast", "person", "tool"]  # Reihenfolge
MAX_FOLLOWUPS_PER_TOPIC = 2

# =======================
# THREAD-SAFE BACKGROUND STORES (kein st.session_state in Threads!)
# =======================
BG_TASKS: Dict[str, Dict[str, Any]] = {}   # task_id -> {status, kind, topic_key, result/error}
BG_RESULTS: List[Dict[str, Any]] = []      # fertige Resultate, wird im Main-Thread geharvestet
BG_LOCK = threading.Lock()

def bg_set_task(task_id: str, data: Dict[str, Any]):
    with BG_LOCK:
        BG_TASKS[task_id] = data

def bg_append_result(res: Dict[str, Any]):
    with BG_LOCK:
        BG_RESULTS.append(res)

def bg_get_and_clear_results() -> List[Dict[str, Any]]:
    with BG_LOCK:
        items = list(BG_RESULTS)
        BG_RESULTS.clear()
        return items

def bg_get_tasks_snapshot() -> Dict[str, Dict[str, Any]]:
    with BG_LOCK:
        return dict(BG_TASKS)

# =======================
# SESSION / STATE
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
        st.session_state.profile = {"topics": []}  # siehe Topic-Objekte
    if "topic_state" not in st.session_state:
        st.session_state.topic_state = {
            "current": "book",
            "followups": {k: 0 for k in TOPIC_ORDER}
        }
    if "inflight_keys" not in st.session_state:
        st.session_state.inflight_keys = set()  # zur Deduplizierung paralleler Aktionen
    if "results" not in st.session_state:
        st.session_state.results: List[Dict[str, Any]] = []
    if "last_director_json" not in st.session_state:
        st.session_state.last_director_json = None
    if "last_raw_director_text" not in st.session_state:
        st.session_state.last_raw_director_text = ""
    if "SHOW_DEBUG" not in st.session_state:
        st.session_state.SHOW_DEBUG = False

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
        "GOALS by topic:\n"
        "- book: title (required), optional author_guess; why (1–2 sentences)\n"
        "- podcast: title; why (1–2 sentences)\n"
        "- role_model (person): name; why (1–2 sentences)\n"
        "- tool: title; why (1–2 sentences)\n"
        "BEHAVIOR:\n"
        "- Keep conversation warm, natural; ask targeted follow-ups that reveal the user's thinking (no redundant confirmations).\n"
        "- If title/name is sufficiently unambiguous (or author provided), you MAY trigger an image action immediately.\n"
        "- If multiple likely authors exist, ask ONE brief disambiguation question.\n"
        "- After ~1–2 meaningful follow-ups per topic, move to the next topic (book → podcast → role_model → tool).\n"
        "CONTROL OUTPUT (STRICT JSON). Output ONLY the JSON object:\n"
        "{\n"
        "  assistant_visible: string,\n"
        "  topic_focus?: 'book'|'podcast'|'person'|'tool',\n"
        "  facts_partial: {\n"
        "    book?: {title?, author_guess?, why?},\n"
        "    podcast?: {title?, why?},\n"
        "    role_model?: {name?, why?},\n"
        "    tool?: {title?, why?},\n"
        "    extras?: object\n"
        "  },\n"
        "  actions: [\n"
        "    { type:'search_cover',    kind:'book|podcast', title?:string, author?:string },\n"
        "    { type:'search_portrait', kind:'person',       name?:string },\n"
        "    { type:'search_logo',     kind:'tool',         title?:string }\n"
        "  ],\n"
        "  next_topic_hint?: 'book'|'podcast'|'person'|'tool'\n"
        "}\n"
    )

def director_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "assistant_visible": {"type": "string"},
            "topic_focus": {"type": "string", "enum": ["book", "podcast", "person", "tool"]},
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
                        "type": {"type": "string", "enum": ["search_cover", "search_portrait", "search_logo"]},
                        "kind": {"type": "string", "enum": ["book", "podcast", "person", "tool"]},
                        "title": {"type": "string"},
                        "author": {"type": "string"},
                        "name": {"type": "string"}
                    },
                    "required": ["type", "kind"]
                }
            },
            "next_topic_hint": {"type": "string", "enum": ["book", "podcast", "person", "tool"]}
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
        return base + "\nEnsure the images are BOOK COVERS (portrait aspect, visible title/author typical of covers)."
    if kind == "podcast":
        return base + "\nFind PODCAST/YOUTUBE CHANNEL ARTWORK (logo/title)."
    if kind == "person":
        return base + "\nFind PORTRAITS (press photos, Wikipedia portrait)."
    if kind == "tool":
        return base + "\nFind LOGOS or official product images."
    return base

def validator_system_prompt(kind: str) -> str:
    return (
        "You are a VISION validator.\n"
        "Inputs:\n"
        "- expected strings (title/author/name),\n"
        "- up to 3 candidate images (as image inputs) with metadata (page_url, source, scraped title/authors if available).\n"
        "Task:\n"
        f"- Inspect each image VISUALLY to judge whether it is a plausible {kind} image.\n"
        "- Verify if visible title/author/name aligns with expected strings (minor differences are fine).\n"
        "- Prefer trustworthy sources (publisher, Amazon, Goodreads, Open Library, official site) if similar.\n"
        "Output STRICT JSON: {best_index: 0|1|2|-1, reason: string, confidence: number}. No prose."
    )

def finalizer_system_prompt() -> str:
    return (
        "You are a finalizing writer. Create a professional, warm, uplifting 3–5 sentence paragraph that presents the user positively without flattery. "
        "Use the interview facts (why the item matters) and the verified media (title/author/name). Be specific, concise, and human."
    )

# =======================
# JSON PARSER (robust)
# =======================
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
def strip_code_fences(s: str) -> str:
    return CODE_FENCE_RE.sub("", s).strip()
def extract_balanced_json(s: str) -> Optional[str]:
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
    topic = {"kind": kind, "facts": {key_field: key_value}, "status": "collecting", "media": {}}
    topics.append(topic)
    return len(topics) - 1

def profile_key(profile: Dict[str, Any], kind: str) -> str:
    for t in profile["topics"]:
        if t["kind"] == kind:
            key_field = "name" if kind == "person" else "title"
            return t.get("facts", {}).get(key_field, "")
    return ""

def merge_facts(profile: Dict[str, Any], partial: Dict[str, Any]):
    if not partial:
        return
    if "book" in partial and isinstance(partial["book"], dict):
        title = partial["book"].get("title", "") or profile_key(profile, "book")
        idx = ensure_topic(profile, "book", title)
        profile["topics"][idx]["facts"].update({k: v for k, v in partial["book"].items() if isinstance(v, str) and v.strip()})
    if "podcast" in partial and isinstance(partial["podcast"], dict):
        title = partial["podcast"].get("title", "") or profile_key(profile, "podcast")
        idx = ensure_topic(profile, "podcast", title)
        profile["topics"][idx]["facts"].update({k: v for k, v in partial["podcast"].items() if isinstance(v, str) and v.strip()})
    if "role_model" in partial and isinstance(partial["role_model"], dict):
        name = partial["role_model"].get("name", "") or profile_key(profile, "person")
        idx = ensure_topic(profile, "person", name)
        profile["topics"][idx]["facts"].update({k: v for k, v in partial["role_model"].items() if isinstance(v, str) and v.strip()})
    if "tool" in partial and isinstance(partial["tool"], dict):
        title = partial["tool"].get("title", "") or profile_key(profile, "tool")
        idx = ensure_topic(profile, "tool", title)
        profile["topics"][idx]["facts"].update({k: v for k, v in partial["tool"].items() if isinstance(v, str) and v.strip()})

# =======================
# AGENT CALLS
# =======================
def call_director(client: OpenAI, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = {
        "model": CHAT_MODEL_PRIMARY,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "director_control", "schema": director_schema(), "strict": True}},
        "input": [
            {"role": "system", "content": system_director_prompt()},
            *history[2:]
        ]
    }
    last_err = None
    for m in [CHAT_MODEL_PRIMARY, *CHAT_MODEL_FALLBACKS]:
        payload["model"] = m
        try:
            r = responses_create(client, **payload)
            text = getattr(r, "output_text", "")
            if not text and getattr(r, "output", None):
                blk = r.output[0]
                if blk and blk.get("content") and blk["content"][0].get("text"):
                    text = blk["content"][0]["text"]
            st.session_state.last_raw_director_text = text or ""
            data, _ = parse_json_strict_or_best_effort(text or "")
            if data is not None:
                return data
            # Fallback neutral (zählt NICHT als Follow-up)
            return {"assistant_visible": "Could you share what, in one or two sentences, resonates most with you?",
                    "facts_partial": {}, "actions": []}
        except Exception as e:
            last_err = e
            continue
    st.session_state.last_raw_director_text = f"(error: {last_err})" if last_err else ""
    return {"assistant_visible": "What stands out to you briefly?", "facts_partial": {}, "actions": []}

def call_websearch(client: OpenAI, kind: str, title_or_name: str, author: str = "") -> Dict[str, Any]:
    payload = {
        "model": SEARCH_MODEL_PRIMARY,
        "tools": [{"type": "web_search"}],
        "response_format": {"type": "json_object"},
        "input": [
            {"role": "system", "content": websearch_system_prompt(kind)},
            {"role": "user", "content": json.dumps({
                "kind": kind, "title_or_name": title_or_name, "author": author
            }, separators=(",", ":"))}
        ]
    }
    for m in [SEARCH_MODEL_PRIMARY, *SEARCH_MODEL_FALLBACKS]:
        payload["model"] = m
        try:
            r = responses_create(client, **payload)
            text = getattr(r, "output_text", "") or (r.output[0]["content"][0]["text"] if getattr(r, "output", None) else "")
            data, _ = parse_json_strict_or_best_effort(text or "")
            cands = (data or {}).get("candidates") or []
            return {"candidates": cands[:3]}
        except Exception:
            continue
    return {"candidates": []}

def call_vision_validator(client: OpenAI, kind: str,
                          expected_title: str, expected_author_or_name: str,
                          candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = [
        {"type": "input_text",
         "text": json.dumps({
             "kind": kind,
             "expected_title": expected_title,
             "expected_author_or_name": expected_author_or_name,
             "candidates_meta": [
                 {"page_url": c.get("page_url", ""), "source": c.get("source", ""),
                  "title": c.get("title", ""), "authors": c.get("authors", []),
                  "image_url": c.get("image_url", "")}
                 for c in candidates[:3]
             ]
         }, separators=(",", ":"))}
    ]
    for c in candidates[:3]:
        if c.get("image_url"):
            parts.append({"type": "input_image", "image_url": c["image_url"]})

    payload = {
        "model": CHAT_MODEL_PRIMARY,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "validation",
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
            {"role": "user", "content": parts}
        ]
    }
    for m in [CHAT_MODEL_PRIMARY, *CHAT_MODEL_FALLBACKS]:
        payload["model"] = m
        try:
            r = responses_create(client, **payload)
            text = getattr(r, "output_text", "") or (r.output[0]["content"][0]["text"] if getattr(r, "output", None) else "")
            data, _ = parse_json_strict_or_best_effort(text or "")
            if data is None:
                return {"best_index": -1, "reason": "Parse failed", "confidence": 0.0}
            return data
        except Exception:
            continue
    return {"best_index": -1, "reason": "Validator failed", "confidence": 0.0}

def call_finalizer(client: OpenAI, facts: Dict[str, Any], chosen_meta: Dict[str, Any]) -> str:
    payload = {
        "model": CHAT_MODEL_PRIMARY,
        "temperature": 0.6,
        "input": [
            {"role": "system", "content": finalizer_system_prompt()},
            {"role": "user", "content": json.dumps({"facts": facts, "verified_media": chosen_meta}, separators=(",", ":"))}
        ]
    }
    for m in [CHAT_MODEL_PRIMARY, *CHAT_MODEL_FALLBACKS]:
        payload["model"] = m
        try:
            r = responses_create(client, **payload)
            txt = getattr(r, "output_text", "") or (r.output[0]["content"][0]["text"] if getattr(r, "output", None) else "")
            return (txt or "").strip()
        except Exception:
            continue
    return "A concise, positive summary will appear here once the network recovers."

# =======================
# DOWNLOADER (Anzeige)
# =======================
def safe_download_image(url: str) -> Optional[str]:
    try:
        with requests.get(url, timeout=HTTP_TIMEOUT, stream=True, headers={"User-Agent": "Mozilla/5.0"}) as r:
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if ctype not in ALLOWED_IMAGE_MIME:
                return None
            suf = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(ctype, ".img")
            fd, path = tempfile.mkstemp(prefix="img_", suffix=suf)
            with os.fdopen(fd, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return path
    except Exception:
        return None

# =======================
# BACKGROUND TASKS (ohne st.session_state!)
# =======================
def start_media_task(client: OpenAI, action_type: str, kind: str,
                     title_or_name: str, author: str, topic_key: str, facts_snapshot: Dict[str, Any]):
    """
    topic_key: stabiler Schlüssel z. B. f"{kind}:{title_or_name or name}". Dient zum Zuordnen im Harvest.
    """
    task_id = str(uuid.uuid4())
    bg_set_task(task_id, {"status": "running", "kind": action_type, "topic_key": topic_key, "error": "", "result": None})

    def worker():
        try:
            # 1) Websuche
            found = call_websearch(client, kind, title_or_name, author)
            candidates = (found.get("candidates") or [])[:3]

            # 2) Vision-Validierung
            expected_title = title_or_name if kind in ("book", "podcast", "tool") else ""
            expected_author_or_name = author if kind == "book" else (title_or_name if kind == "person" else "")
            val = call_vision_validator(client, kind, expected_title, expected_author_or_name, candidates)
            best_i = val.get("best_index", -1)
            chosen = candidates[best_i] if 0 <= best_i < len(candidates) else {}

            # 3) Finalizer
            final_txt = call_finalizer(client, facts_snapshot, chosen)

            res = {
                "topic_key": topic_key,
                "kind": kind,
                "image": {
                    "image_url": chosen.get("image_url"),
                    "page_url": chosen.get("page_url"),
                    "source": chosen.get("source")
                },
                "text": {"final_paragraph": final_txt},
                "meta": {
                    "validator": val,
                    "candidates": candidates
                }
            }
            bg_append_result(res)
            bg_set_task(task_id, {"status": "done", "kind": action_type, "topic_key": topic_key, "error": "", "result": res})
        except Exception as e:
            bg_set_task(task_id, {"status": "error", "kind": action_type, "topic_key": topic_key, "error": str(e), "result": None})

    threading.Thread(target=worker, daemon=True).start()

# =======================
# TOPIC MANAGER
# =======================
def current_topic() -> str:
    return st.session_state.topic_state["current"]

def inc_followup(kind: str):
    st.session_state.topic_state["followups"][kind] = st.session_state.topic_state["followups"].get(kind, 0) + 1

def followups(kind: str) -> int:
    return st.session_state.topic_state["followups"].get(kind, 0)

def switch_to_next_topic(hint: Optional[str] = None):
    order = TOPIC_ORDER
    cur = current_topic()
    if hint in order:
        st.session_state.topic_state["current"] = hint
        return
    idx = order.index(cur) if cur in order else -1
    nxt = order[idx + 1] if 0 <= idx < len(order) - 1 else order[0]
    st.session_state.topic_state["current"] = nxt

def maybe_switch_topic(hint: Optional[str] = None):
    cur = current_topic()
    if followups(cur) >= MAX_FOLLOWUPS_PER_TOPIC:
        switch_to_next_topic(hint)

# Fakten-Delta (nur dann Follow-up zählen)
def facts_delta_count(before: Dict[str, Any], after: Dict[str, Any], topic_kind: str) -> int:
    key_field = "name" if topic_kind == "person" else "title"
    b = before.get(topic_kind, {})
    a = after.get(topic_kind, {})
    keys = set(b.keys()) | set(a.keys())
    added = 0
    for k in keys:
        bv = (b.get(k) or "").strip() if isinstance(b.get(k), str) else b.get(k)
        av = (a.get(k) or "").strip() if isinstance(a.get(k), str) else a.get(k)
        if (not bv) and av:
            added += 1
        # Änderung von vorhandenem Text auf besseren längeren Text zählen wir nicht doppelt
    return added

# =======================
# STREAMLIT UI
# =======================
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="centered")
colA, colB = st.columns([1, 1])
with colA:
    st.title("Dein Kurz-Steckbrief")
    st.caption("Director decides • parallel search + vision • final text per item")
with colB:
    st.toggle("Debug", key="SHOW_DEBUG", help="Show raw JSON/text outputs for troubleshooting")

ensure_session()

# Harvest BG results (MAIN-THREAD)
harvested = bg_get_and_clear_results()
for res in harvested:
    # Map auf Topic im Session-Profile
    topic_key = res.get("topic_key", "")
    # topic_key ist "kind:keyvalue"
    try:
        kind, keyvalue = topic_key.split(":", 1)
    except Exception:
        continue
    # ensure topic
    idx = ensure_topic(st.session_state.profile, kind, keyvalue)
    topic = st.session_state.profile["topics"][idx]
    topic["status"] = "ready"
    topic.setdefault("media", {})
    topic["media"]["candidates"] = res.get("meta", {}).get("candidates", [])
    topic["media"]["validator"] = res.get("meta", {}).get("validator", {})
    topic["media"]["chosen"] = {
        "image_url": res.get("image", {}).get("image_url"),
        "page_url": res.get("image", {}).get("page_url"),
        "source": res.get("image", {}).get("source"),
    }
    # Results-Liste für Darstellung
    st.session_state.results.append(res)
    # inflight entfernen
    st.session_state.inflight_keys.discard(topic_key)

# Fortschritt/Badges je Topic
st.subheader("Topics")
badge_cols = st.columns(len(TOPIC_ORDER))
for i, kind in enumerate(TOPIC_ORDER):
    status = "—"
    for t in st.session_state.profile["topics"]:
        if t["kind"] == kind:
            status = t.get("status", "collecting")
            break
    with badge_cols[i]:
        cur_marker = " (current)" if current_topic() == kind else ""
        st.markdown(f"**{kind.capitalize()}**{cur_marker}")
        st.caption(f"status: {status} • followups: {followups(kind)}")

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
        # Snapshot vor Director (für Delta-Berechnung)
        before_partial = {"book": {}, "podcast": {}, "role_model": {}, "tool": {}}

        director_json = call_director(client, st.session_state.messages)
        st.session_state.last_director_json = director_json

        # Facts mergen
        facts_partial = director_json.get("facts_partial") or {}
        # für Delta braucht man per-kind Blöcke (nur die, die gekommen sind)
        for k in before_partial.keys():
            if k in facts_partial and isinstance(facts_partial[k], dict):
                before_partial[k] = {}

        # Alte Facts snapshotten für Delta-Vergleich:
        current_kind = director_json.get("topic_focus") or current_topic()
        old_block = {}
        # hole alten Block
        for t in st.session_state.profile["topics"]:
            if t["kind"] == current_kind:
                old_block = dict(t.get("facts", {}))
                break

        merge_facts(st.session_state.profile, facts_partial)

        # Sichtbare Antwort
        vis = (director_json.get("assistant_visible") or "").strip()
        if vis:
            st.session_state.messages.append({"role": "assistant", "content": vis})

        # Delta prüfen (nur dann Follow-up erhöhen)
        # hole neuen Block
        new_block = {}
        for t in st.session_state.profile["topics"]:
            if t["kind"] == current_kind:
                new_block = dict(t.get("facts", {}))
                break
        # baue einfache Maps fürs Delta
        delta_before = {current_kind: old_block}
        delta_after = {current_kind: new_block}
        if facts_delta_count(delta_before, delta_after, current_kind) > 0:
            st.session_state.topic_state["followups"][current_kind] = st.session_state.topic_state["followups"].get(current_kind, 0) + 1

        # Aktionen ausführen (parallel, dedupliziert)
        for act in (director_json.get("actions") or []):
            a_type = act.get("type", "")
            kind = (act.get("kind") or "").strip()
            if not a_type or not kind:
                continue

            if kind == "person":
                name = (act.get("name") or profile_key(st.session_state.profile, "person")).strip()
                if not name:
                    continue
                topic_key = f"{kind}:{name}"
                if topic_key in st.session_state.inflight_keys:
                    continue
                st.session_state.inflight_keys.add(topic_key)
                snapshot = {"facts": {"name": name}}
                start_media_task(client, a_type, "person", name, "", topic_key, snapshot)
                # markiere Topic als queued
                idx = ensure_topic(st.session_state.profile, "person", name)
                st.session_state.profile["topics"][idx]["status"] = "queued"
            else:
                title = (act.get("title") or profile_key(st.session_state.profile, kind)).strip()
                author = (act.get("author") or "").strip()
                if not title:
                    continue
                topic_key = f"{kind}:{title}"
                if topic_key in st.session_state.inflight_keys:
                    continue
                st.session_state.inflight_keys.add(topic_key)
                snapshot = {"facts": {"title": title, "author_guess": author} if author else {"title": title}}
                start_media_task(client, a_type, kind, title, author, topic_key, snapshot)
                # markiere Topic als queued
                idx = ensure_topic(st.session_state.profile, kind, title)
                st.session_state.profile["topics"][idx]["status"] = "queued"

        # Topic-Wechsel (nur wenn Follow-up-Limit erreicht ODER der Director explizit hintet)
        maybe_switch_topic(director_json.get("next_topic_hint"))

    else:
        st.session_state.messages.append({"role": "assistant",
                                          "content": "(Demo mode — set OPENAI_API_KEY in Streamlit Secrets.)"})
    st.rerun()

# Laufende Tasks
tasks_snapshot = bg_get_tasks_snapshot()
running = [t for t in tasks_snapshot.values() if t["status"] == "running"]
if running or st.session_state.inflight_keys:
    st.info("🔎 Background research running… you can keep chatting.")
    if st.button("🔄 Refresh results"):
        st.rerun()

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
            st.markdown(f"**{kind.capitalize() if kind else 'Item'}**")
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
        for k in ["messages", "profile", "results", "last_director_json", "last_raw_director_text", "topic_state", "inflight_keys"]:
            if k in st.session_state:
                del st.session_state[k]
        ensure_session()
        # globale Stores auch leeren
        with BG_LOCK:
            BG_TASKS.clear(); BG_RESULTS.clear()
        st.rerun()
with c2:
    if st.session_state.SHOW_DEBUG and st.session_state.get("last_director_json") is not None:
        with st.expander("Last director JSON (parsed)"):
            st.code(json.dumps(st.session_state.last_director_json, indent=2))
with c3:
    if st.session_state.SHOW_DEBUG and st.session_state.get("last_raw_director_text"):
        with st.expander("Last director RAW text"):
            st.code(st.session_state.last_raw_director_text)
