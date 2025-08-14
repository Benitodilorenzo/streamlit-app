# app.py — Expert Card Multi-Agent (Streamlit + OpenAI Responses + Vision)
# Director-Chat führt natürliches Gespräch (ohne redundante Bestätigungen),
# triggert eigenständig Actions (z. B. search_cover).
# Hintergrund-Pipeline: Websuche -> Vision-Validierung -> Finalizer.
# Ausgabe: Zweispalten-Layout (Bild links, Text rechts).

import os
import io
import json
import uuid
import threading
import tempfile
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI
from PIL import Image
import requests

# =======================
# CONFIG
# =======================
APP_TITLE = "Expert Card – Mini Agents (Vision)"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Basis-Modelle
CHAT_MODEL_PRIMARY = os.getenv("DTBR_CHAT_MODEL", "gpt-5")            # Gespräch, Controller, Finalizer, Vision-Validator
CHAT_MODEL_FALLBACKS = ["gpt-4o", "gpt-4o-mini"]

SEARCH_MODEL_PRIMARY = os.getenv("DTBR_SEARCH_MODEL", "gpt-5")        # mit web_search Tool
SEARCH_MODEL_FALLBACKS = ["gpt-4.1", "gpt-4o"]

# Downloads
ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/webp"}
HTTP_TIMEOUT = 20

SHOW_DEBUG = False  # True -> Debugblöcke

# =======================
# SESSION
# =======================
def ensure_session():
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_director_prompt()},
            {"role": "assistant", "content": (
                "Hi! I'll help you craft a tiny expert card. "
                "First, which book has helped you professionally? "
                "Please give the title (author optional)."
            )},
        ]
    if "facts" not in st.session_state:
        # Platz für mehr als nur Buch: Podcasts, Role Model etc. (optional)
        st.session_state.facts = {
            "book": {"title": "", "author_guess": "", "why": ""}
        }
    if "tasks" not in st.session_state:
        st.session_state.tasks: Dict[str, Dict[str, Any]] = {}  # task_id -> dict(status, result, error, kind)
    if "results" not in st.session_state:
        st.session_state.results: List[Dict[str, Any]] = []     # fertige Ergebnisse (Bild + Finalizer-Text)
    if "last_director_json" not in st.session_state:
        st.session_state.last_director_json = None

def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set (Streamlit → Settings → Secrets).")
    return OpenAI(api_key=OPENAI_API_KEY)

# =======================
# PROMPTS
# =======================
def system_director_prompt() -> str:
    """
    Chat-Agent (Director):
    - führt natürliches, freundliches Gespräch (Default Englisch; spiegle Nutzersprache, wenn eindeutig nicht Englisch),
    - sammelt mind. Buch.Titel; optional Autor; und eine kurze Begründung (1–2 Sätze),
    - KEINE redundanten Bestätigungen („Is that correct?“ etc.),
    - wenn ausreichend eindeutig: ohne Nachfrage Action `search_cover` starten,
    - bei Mehrdeutigkeit: genau EINE knappe Disambiguation-Frage,
    - gerne weitere, leichte Folgefragen (Podcasts, Tipps etc.) – aber nie den Nutzer „zuwortfragen“,
    - UI erwähnt keine Hintergrundarbeit.
    Ausgabe IMMER als JSON gemäß Schema.
    """
    return (
        "You are the Interview Director for a tiny 'Expert Card'.\n"
        "LANGUAGE:\n"
        "- Default to English; mirror the user's language if clearly not English.\n"
        "GOALS:\n"
        "- Collect book.title (required), optionally book.author_guess, and a short book.why (1–2 sentences).\n"
        "- Keep conversation natural and warm; brief follow-ups are welcome (e.g., podcasts, role models, tips).\n"
        "BEHAVIOR:\n"
        "- NEVER ask redundant confirmations if info is already clear.\n"
        "- If title is present and either author is present OR the title is sufficiently unambiguous, you MAY trigger a cover search.\n"
        "- If multiple likely authors exist, ask ONE targeted disambiguation question.\n"
        "- Keep messages short, friendly, and specific. Avoid over-politeness.\n"
        "OUTPUT (STRICT JSON):\n"
        "- Return an object with:\n"
        "  assistant_visible: string (next message to user, in user's language),\n"
        "  facts_partial: { book: { title?, author_guess?, why? }, extras?: any },\n"
        "  actions: [ { type: 'search_cover', title: string, author?: string } ] or [],\n"
        "  notes: string (optional internal rationale; ignored by UI).\n"
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
                    "extras": {"type": "object"}
                }
            },
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["search_cover"]},
                        "title": {"type": "string"},
                        "author": {"type": "string"}
                    },
                    "required": ["type", "title"]
                }
            },
            "notes": {"type": "string"}
        },
        "required": ["assistant_visible", "facts_partial", "actions"]
    }

def websearch_system_prompt() -> str:
    return (
        "You are a web image search assistant.\n"
        "Find up to three HIGH-QUALITY BOOK COVER candidates.\n"
        "Return ONLY JSON with: candidates: [ {image_url, page_url, source, title, authors[]} ].\n"
        "Prefer trusted sources: publisher, Amazon, Goodreads, Open Library, Wikipedia/Commons.\n"
        "Whenever possible, provide DIRECT image URLs (not thumbnails that require JS)."
    )

def validator_system_prompt() -> str:
    return (
        "You are a VISION validator.\n"
        "Inputs:\n"
        "- expected title/author (strings),\n"
        "- up to 3 candidate images (as image inputs) with metadata (page_url, source, scraped title/authors if available).\n"
        "Task:\n"
        "- Inspect each image VISUALLY to judge whether it is a plausible book cover.\n"
        "- Verify if title/author appear consistent with the expected strings (tolerate small punctuation/case variations).\n"
        "- Prefer trustworthy sources (publisher, Amazon, Goodreads, Open Library) if images look similar.\n"
        "Output STRICT JSON: {best_index: 0|1|2|-1, reason: string}."
    )

def finalizer_system_prompt() -> str:
    return (
        "You are a finalizing writer. Create a professional, warm, uplifting 3–5 sentence paragraph that presents the user positively without flattery.\n"
        "Use the interview facts and the verified book (title/author) and why it helped. Be specific, concise, and human."
    )

# =======================
# OPENAI WRAPPERS
# =======================
def responses_create(client: OpenAI, **kwargs):
    try:
        return client.responses.create(**kwargs)
    except TypeError:
        cleaned = dict(kwargs)
        cleaned.pop("response_format", None)
        cleaned.pop("tools", None)
        return client.responses.create(**cleaned)

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
            *history[2:]  # skip initial system+assistant boot messages already in history
        ]
    }
    last = None
    for m in [CHAT_MODEL_PRIMARY, *CHAT_MODEL_FALLBACKS]:
        payload["model"] = m
        try:
            r = responses_create(client, **payload)
            text = getattr(r, "output_text", "")
            if not text and getattr(r, "output", None):
                blk = r.output[0]
                if blk and blk.get("content") and blk["content"][0].get("text"):
                    text = blk["content"][0]["text"]
            return json.loads(text)
        except Exception as e:
            last = e
            continue
    raise last

def call_websearch(client: OpenAI, title: str, author: str) -> Dict[str, Any]:
    payload = {
        "model": SEARCH_MODEL_PRIMARY,
        "tools": [{"type": "web_search"}],  # Responses + Web Search Tool
        "response_format": {"type": "json_object"},
        "input": [
            {"role": "system", "content": websearch_system_prompt()},
            {"role": "user", "content": json.dumps({"book_title": title, "author": author}, separators=(",", ":"))}
        ]
    }
    last = None
    for m in [SEARCH_MODEL_PRIMARY, *SEARCH_MODEL_FALLBACKS]:
        payload["model"] = m
        try:
            r = responses_create(client, **payload)
            text = getattr(r, "output_text", "") or (r.output[0]["content"][0]["text"] if getattr(r, "output", None) else "")
            data = json.loads(text) if text else {}
            return {"candidates": (data.get("candidates") or [])[:3]}
        except Exception as e:
            last = e
            continue
    raise last

def call_vision_validator(client: OpenAI, expected_title: str, expected_author: str,
                          candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Vision-Validierung: wir geben die Bilder direkt als Bild-Inputs in die Responses API.
    """
    # Baue einen Input-Block mit Text + bis zu 3 "input_image" Content-Parts
    content_parts: List[Dict[str, Any]] = [
        {"type": "input_text",
         "text": json.dumps({
             "expected_title": expected_title,
             "expected_author": expected_author,
             "candidates_meta": [
                 {"page_url": c.get("page_url", ""), "source": c.get("source", ""),
                  "title": c.get("title", ""), "authors": c.get("authors", [])}
                 for c in candidates[:3]
             ]
         }, separators=(",", ":"))}
    ]

    for c in candidates[:3]:
        img_url = c.get("image_url")
        if img_url:
            content_parts.append({"type": "input_image", "image_url": img_url})

    payload = {
        "model": CHAT_MODEL_PRIMARY,  # Vision-fähig
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "cover_validation",
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "best_index": {"type": "integer"},
                                                    "reason": {"type": "string"}
                                                },
                                                "required": ["best_index", "reason"]
                                            },
                                            "strict": True}},
        "input": [
            {"role": "system", "content": validator_system_prompt()},
            {"role": "user", "content": content_parts}
        ]
    }
    last = None
    for m in [CHAT_MODEL_PRIMARY, *CHAT_MODEL_FALLBACKS]:
        payload["model"] = m
        try:
            r = responses_create(client, **payload)
            text = getattr(r, "output_text", "") or (r.output[0]["content"][0]["text"] if getattr(r, "output", None) else "")
            return json.loads(text) if text else {"best_index": -1, "reason": "No output"}
        except Exception as e:
            last = e
            continue
    raise last

def call_finalizer(client: OpenAI, facts: Dict[str, Any], chosen: Dict[str, Any]) -> str:
    payload = {
        "model": CHAT_MODEL_PRIMARY,
        "input": [
            {"role": "system", "content": finalizer_system_prompt()},
            {"role": "user", "content": json.dumps({"facts": facts, "verified_book": chosen}, separators=(",", ":"))}
        ]
    }
    last = None
    for m in [CHAT_MODEL_PRIMARY, *CHAT_MODEL_FALLBACKS]:
        payload["model"] = m
        try:
            r = responses_create(client, **payload)
            txt = getattr(r, "output_text", "") or (r.output[0]["content"][0]["text"] if getattr(r, "output", None) else "")
            return (txt or "").strip()
        except Exception as e:
            last = e
            continue
    raise last

# =======================
# IMAGE DOWNLOAD (für Anzeige)
# =======================
def safe_download_image(url: str) -> Optional[str]:
    try:
        with requests.get(url, timeout=HTTP_TIMEOUT, stream=True, headers={"User-Agent": "Mozilla/5.0"}) as r:
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if ctype not in ALLOWED_IMAGE_MIME:
                return None
            fd, path = tempfile.mkstemp(prefix="cover_", suffix={
                "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"
            }.get(ctype, ".img"))
            with os.fdopen(fd, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return path
    except Exception:
        return None

# =======================
# BACKGROUND TASK: COVER SEARCH → VALIDATE (VISION) → FINALIZE
# =======================
def start_cover_task(client: OpenAI, title: str, author: str, facts_snapshot: Dict[str, Any]):
    task_id = str(uuid.uuid4())
    st.session_state.tasks[task_id] = {"status": "running", "error": "", "result": None, "kind": "cover"}

    def worker():
        try:
            # 1) Websuche (bis zu 3 Kandidaten)
            found = call_websearch(client, title, author)
            candidates = (found.get("candidates") or [])[:3]

            # 2) Vision-Validierung
            val = call_vision_validator(client, title, author, candidates)
            best_i = val.get("best_index", -1)
            reason = val.get("reason", "")

            chosen = candidates[best_i] if 0 <= best_i < len(candidates) else {}
            img_path = None
            if chosen.get("image_url"):
                img_path = safe_download_image(chosen["image_url"])

            # 3) Finalizer-Text
            final_txt = call_finalizer(client, facts_snapshot, chosen)

            res = {
                "title": title,
                "author": author,
                "chosen_meta": chosen,
                "validator_reason": reason,
                "final_text": final_txt,
                "local_image": img_path,
                "all_candidates": candidates
            }
            st.session_state.tasks[task_id] = {"status": "done", "error": "", "result": res, "kind": "cover"}
            st.session_state.results.append(res)
        except Exception as e:
            st.session_state.tasks[task_id] = {"status": "error", "error": str(e), "result": None, "kind": "cover"}

    threading.Thread(target=worker, daemon=True).start()

# =======================
# STREAMLIT UI
# =======================
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="centered")
st.title("Dein Kurz-Steckbrief")
st.caption("Director decides; background cover search with Vision; final shows as two columns")

ensure_session()

# Verlauf anzeigen (ohne initiales system)
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

# Chat input
user_text = st.chat_input("Type your answer…")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    if agent_ready:
        # Director (ein Call, strukturiert)
        dj = call_director(client, st.session_state.messages)
        st.session_state.last_director_json = dj

        # Fakten mergen
        facts_partial = dj.get("facts_partial") or {}
        bp = (facts_partial.get("book") or {})
        for k, v in bp.items():
            if isinstance(v, str) and not v.strip():
                continue
            st.session_state.facts["book"][k] = v

        # Sichtbarer Bot-Text
        vis = (dj.get("assistant_visible") or "").strip()
        if vis:
            st.session_state.messages.append({"role": "assistant", "content": vis})

        # Aktionen
        for act in (dj.get("actions") or []):
            if act.get("type") == "search_cover":
                title = (act.get("title") or st.session_state.facts["book"].get("title", "")).strip()
                author = (act.get("author") or st.session_state.facts["book"].get("author_guess", "")).strip()
                if title:
                    snapshot = json.loads(json.dumps(st.session_state.facts))
                    start_cover_task(client, title, author, snapshot)

    else:
        st.session_state.messages.append({"role": "assistant",
                                          "content": "(Demo mode — set OPENAI_API_KEY in Streamlit Secrets.)"})
    st.rerun()

# Status laufender Tasks
running = [t for t in st.session_state.tasks.values() if t["status"] == "running"]
if running:
    st.info("🔎 Background research running… feel free to continue the chat.")

# Ergebnisse anzeigen
if st.session_state.results:
    st.subheader("Results")
    for res in st.session_state.results[::-1]:
        c1, c2 = st.columns([1, 1.25])
        with c1:
            if res.get("local_image") and os.path.exists(res["local_image"]):
                st.image(res["local_image"], caption=res.get("chosen_meta", {}).get("source", ""), use_container_width=True)
            elif res.get("chosen_meta", {}).get("image_url"):
                # Fallback: zeige Remote-URL (Streamlit kann das direkt)
                st.image(res["chosen_meta"]["image_url"], caption=res.get("chosen_meta", {}).get("source", ""), use_container_width=True)
            else:
                st.write("No image available.")
        with c2:
            st.markdown(f"**Book:** {res.get('title','')} — {res.get('author','')}")
            st.write(res.get("final_text", ""))
            cm = res.get("chosen_meta") or {}
            if cm.get("page_url"):
                st.markdown(f"[Source]({cm['page_url']})")
            if SHOW_DEBUG:
                with st.expander("Debug (metadata)"):
                    st.code(json.dumps(res, indent=2))

# Controls
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🔄 Restart interview"):
        for k in ["messages", "facts", "tasks", "results", "last_director_json"]:
            if k in st.session_state:
                del st.session_state[k]
        ensure_session()
        st.rerun()
with c2:
    if SHOW_DEBUG and st.session_state.get("last_director_json"):
        with st.expander("Last director JSON"):
            st.code(json.dumps(st.session_state.last_director_json, indent=2))
with c3:
    if st.button("Clear results"):
        st.session_state.results.clear()
        st.rerun()
