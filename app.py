# app.py — Expert Card Multi-Agents (Director + Web Search + Vision + Finalizer)
# v2: Hard follow-up budget, duplicate prompt guard, topic focus, retries/timeouts,
#     thread-safe BG queues, error cards, action dedup, live harvesting.

import os
import re
import json
import time
import uuid
import queue
import threading
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from openai import OpenAI

# =======================
# CONFIG
# =======================
APP_TITLE = "Expert Card – Mini Agents (Vision)"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Models (GPT-5 default, with fallbacks)
CHAT_MODEL_PRIMARY = os.getenv("DTBR_CHAT_MODEL", "gpt-5")
CHAT_MODEL_FALLBACKS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]

SEARCH_MODEL_PRIMARY = os.getenv("DTBR_SEARCH_MODEL", "gpt-5")
SEARCH_MODEL_FALLBACKS = ["gpt-4.1", "gpt-4o", "gpt-4o-mini"]

HTTP_TIMEOUT = 20  # seconds (OpenAI client has its own timeout)
CALL_RETRIES = 2   # additional retries (total attempts = 1 + CALL_RETRIES)
RETRY_BACKOFF = 1.5

TOPIC_ORDER = ["book", "podcast", "person", "tool"]
MAX_FOLLOWUPS_PER_TOPIC = 2

# =======================
# THREAD-SAFE BACKGROUND QUEUES
# =======================
BG_TASKS: Dict[str, Dict[str, Any]] = {}   # id -> {status, kind, topic_key, ...}
BG_RESULTS: "queue.Queue[Dict[str, Any]]" = queue.Queue()
BG_LOCK = threading.Lock()

def bg_set_task(task_id: str, data: Dict[str, Any]):
    with BG_LOCK:
        BG_TASKS[task_id] = data

def bg_finish_task(task_id: str, result: Optional[Dict[str, Any]] = None, error: str = ""):
    with BG_LOCK:
        if task_id in BG_TASKS:
            BG_TASKS[task_id]["status"] = "done" if not error else "error"
            BG_TASKS[task_id]["error"] = error
            if result:
                BG_TASKS[task_id]["result"] = result

def bg_put_result(res: Dict[str, Any]):
    BG_RESULTS.put(res)

def bg_snapshot_tasks() -> Dict[str, Dict[str, Any]]:
    with BG_LOCK:
        return dict(BG_TASKS)

def bg_clear_all():
    with BG_LOCK:
        BG_TASKS.clear()
    while not BG_RESULTS.empty():
        try:
            BG_RESULTS.get_nowait()
        except queue.Empty:
            break

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
        st.session_state.profile = {"topics": []}
    if "topic_state" not in st.session_state:
        st.session_state.topic_state = {
            "current": "book",
            "followups": {k: 0 for k in TOPIC_ORDER}
        }
    if "inflight_keys" not in st.session_state:
        st.session_state.inflight_keys = set()
    if "results" not in st.session_state:
        st.session_state.results: List[Dict[str, Any]] = []
    if "last_assistant_visible" not in st.session_state:
        st.session_state.last_assistant_visible = ""
    if "SHOW_DEBUG" not in st.session_state:
        st.session_state.SHOW_DEBUG = False
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
        "TOPICS (in order): book → podcast → person (role model) → tool.\n"
        "GOALS per topic:\n"
        "- book: title (required), optional author_guess; why (1–2 sentences)\n"
        "- podcast: title; why (1–2 sentences)\n"
        "- person: name; why (1–2 sentences)\n"
        "- tool: title; why (1–2 sentences)\n"
        "BEHAVIOR:\n"
        "- Be warm, focused, no redundant confirmations. Ask targeted follow-ups that reveal the user's thinking.\n"
        "- If title/name is unambiguous (or author provided), you MAY trigger an image action immediately.\n"
        "- If multiple likely authors exist, ask ONE brief disambiguation.\n"
        "- After ~1–2 meaningful follow-ups per topic, MOVE ON to the next topic.\n"
        "CONTROL OUTPUT — STRICT JSON ONLY (no prose):\n"
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
        "    { type:'search_cover',    kind:'book'|'podcast', title?:string, author?:string },\n"
        "    { type:'search_portrait', kind:'person',         name?:string },\n"
        "    { type:'search_logo',     kind:'tool',           title?:string }\n"
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
        "Find up to three HIGH-QUALITY image candidates.\n"
        "Return ONLY JSON: {candidates:[{image_url,page_url,source,title,authors[]}]}.\n"
        "Prefer publisher, Amazon, Goodreads, Open Library, Wikipedia/Commons, or the official site/channel.\n"
        "No extra text; JSON only."
    )
    if kind == "book":
        return base + "\nEnsure images are BOOK COVERS (portrait, typical title/author typography)."
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
        "- Verify if visible title/author/name aligns with expected strings (minor differences OK).\n"
        "- Prefer trustworthy sources if similar.\n"
        "STRICT JSON ONLY: {best_index:0|1|2|-1, reason:string, confidence:number}."
    )

def finalizer_system_prompt() -> str:
    return (
        "You are a finalizing writer. Create a professional, warm, uplifting 3–5 sentence paragraph that presents the user positively without flattery. "
        "Use the interview facts (why the item matters) and the verified media (title/author/name). Be specific, concise, and human."
    )

# =======================
# JSON HELPERS (robust)
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

def parse_json_best_effort(text: str) -> Optional[dict]:
    if not text:
        return None
    for candidate in [text, strip_code_fences(text), extract_balanced_json(strip_code_fences(text))]:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None

# =======================
# OPENAI WRAPPERS (retries, fallbacks)
# =======================
def responses_create(client: OpenAI, model_list: List[str], **kwargs):
    """
    Try models with retries; remove unsupported keys if needed.
    """
    last_err = None
    for m in model_list:
        payload = dict(kwargs)
        payload["model"] = m
        for attempt in range(1 + CALL_RETRIES):
            try:
                return client.responses.create(**payload)
            except Exception as e:
                last_err = e
                # if model/tool combo unsupported → try removing tool/response_format on next try for this model
                if "response_format" in payload and "unsupported" in str(e).lower():
                    payload.pop("response_format", None)
                if "tools" in payload and ("not supported" in str(e).lower() or "model_not_found" in str(e).lower()):
                    payload.pop("tools", None)
                if attempt < CALL_RETRIES:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                else:
                    break
    if last_err:
        raise last_err
    raise RuntimeError("responses_create: no model could be called")

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
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "director_control", "schema": director_schema(), "strict": True}
        },
        "input": [
            {"role": "system", "content": system_director_prompt()},
            *history[2:]  # skip our own system + first assistant seed
        ],
        "temperature": 0.2,
    }
    r = responses_create(client, [CHAT_MODEL_PRIMARY, *CHAT_MODEL_FALLBACKS], **payload)
    text = getattr(r, "output_text", "")
    if not text and getattr(r, "output", None):
        blk = r.output[0]
        if blk and blk.get("content") and blk["content"][0].get("text"):
            text = blk["content"][0]["text"]
    st.session_state.last_raw_director_text = text or ""
    data = parse_json_best_effort(text or "")
    if data is None:
        # neutral fallback (does NOT count as follow-up)
        return {"assistant_visible": "Briefly: what about it resonates most with you?", "facts_partial": {}, "actions": []}
    return data

def call_websearch(client: OpenAI, kind: str, title_or_name: str, author: str = "") -> Dict[str, Any]:
    base_input = [
        {"role": "system", "content": websearch_system_prompt(kind)},
        {"role": "user", "content": json.dumps({"kind": kind, "title_or_name": title_or_name, "author": author}, separators=(",", ":"))}
    ]
    # Strategy A: with web_search tool
    payload_a = {"tools": [{"type": "web_search"}], "response_format": {"type": "json_object"}, "input": base_input, "temperature": 0.1}
    try:
        r = responses_create(client, [SEARCH_MODEL_PRIMARY, *SEARCH_MODEL_FALLBACKS], **payload_a)
        txt = getattr(r, "output_text", "") or (r.output[0]["content"][0]["text"] if getattr(r, "output", None) else "")
        data = parse_json_best_effort(txt or "") or {}
        cands = (data.get("candidates") or [])[:3]
        if cands:
            return {"candidates": cands}
    except Exception:
        pass
    # Strategy B: no tools, ask for links (model knowledge + generic search cues)
    payload_b = {"response_format": {"type": "json_object"}, "input": base_input, "temperature": 0.2}
    try:
        r = responses_create(client, [SEARCH_MODEL_PRIMARY, *SEARCH_MODEL_FALLBACKS], **payload_b)
        txt = getattr(r, "output_text", "") or (r.output[0]["content"][0]["text"] if getattr(r, "output", None) else "")
        data = parse_json_best_effort(txt or "") or {}
        return {"candidates": (data.get("candidates") or [])[:3]}
    except Exception:
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
        ],
        "temperature": 0.1,
    }
    r = responses_create(client, [CHAT_MODEL_PRIMARY, *CHAT_MODEL_FALLBACKS], **payload)
    txt = getattr(r, "output_text", "") or (r.output[0]["content"][0]["text"] if getattr(r, "output", None) else "")
    data = parse_json_best_effort(txt or "")
    if data is None:
        return {"best_index": -1, "reason": "Parse failed", "confidence": 0.0}
    return data

def call_finalizer(client: OpenAI, facts: Dict[str, Any], chosen_meta: Dict[str, Any]) -> str:
    payload = {
        "input": [
            {"role": "system", "content": finalizer_system_prompt()},
            {"role": "user", "content": json.dumps({"facts": facts, "verified_media": chosen_meta}, separators=(",", ":"))}
        ],
        "temperature": 0.6,
    }
    r = responses_create(client, [CHAT_MODEL_PRIMARY, *CHAT_MODEL_FALLBACKS], **payload)
    txt = getattr(r, "output_text", "") or (r.output[0]["content"][0]["text"] if getattr(r, "output", None) else "")
    return (txt or "").strip()

# =======================
# BACKGROUND WORKER (no session access)
# =======================
def start_media_task(client: OpenAI, action_type: str, kind: str,
                     title_or_name: str, author: str, topic_key: str, facts_snapshot: Dict[str, Any]):
    task_id = str(uuid.uuid4())
    bg_set_task(task_id, {"status": "running", "kind": action_type, "topic_key": topic_key, "error": "", "result": None})

    def worker():
        try:
            # 1) Websearch (with retry orchestration is inside wrapper)
            found = call_websearch(client, kind, title_or_name, author)
            candidates = (found.get("candidates") or [])[:3]

            # 2) Vision validate
            expected_title = title_or_name if kind in ("book", "podcast", "tool") else ""
            expected_author_or_name = author if kind == "book" else (title_or_name if kind == "person" else "")
            val = call_vision_validator(client, kind, expected_title, expected_author_or_name, candidates)
            best_i = val.get("best_index", -1)
            chosen = candidates[best_i] if 0 <= best_i < len(candidates) else {}

            # Basic low-confidence retry (one extra search) if needed
            if (best_i < 0 or float(val.get("confidence", 0.0)) < 0.55) and len(candidates) < 3:
                alt = call_websearch(client, kind, title_or_name, author)
                alt_c = (alt.get("candidates") or [])[:3]
                if alt_c:
                    candidates = alt_c
                    val = call_vision_validator(client, kind, expected_title, expected_author_or_name, candidates)
                    best_i = val.get("best_index", -1)
                    chosen = candidates[best_i] if 0 <= best_i < len(candidates) else {}

            # 3) Finalize text
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
                "meta": {"validator": val, "candidates": candidates}
            }
            bg_put_result(res)
            bg_finish_task(task_id, res)
        except Exception as e:
            bg_finish_task(task_id, None, str(e))

    threading.Thread(target=worker, daemon=True).start()

# =======================
# TOPIC MANAGER + GUARDS
# =======================
def current_topic() -> str:
    return st.session_state.topic_state["current"]

def set_current_topic(k: str):
    if k in TOPIC_ORDER:
        st.session_state.topic_state["current"] = k

def followups(kind: str) -> int:
    return st.session_state.topic_state["followups"].get(kind, 0)

def inc_followup(kind: str):
    st.session_state.topic_state["followups"][kind] = followups(kind) + 1

def switch_to_next_topic(hint: Optional[str] = None):
    order = TOPIC_ORDER
    cur = current_topic()
    if hint in order:
        set_current_topic(hint); return
    idx = order.index(cur) if cur in order else -1
    nxt = order[idx + 1] if 0 <= idx < len(order) - 1 else order[0]
    set_current_topic(nxt)

def facts_delta_count(old: Dict[str, Any], new: Dict[str, Any]) -> int:
    added = 0
    for k, v in new.items():
        if isinstance(v, str) and v.strip() and (not isinstance(old.get(k), str) or not old.get(k, "").strip()):
            added += 1
    return added

def collect_topic_block(profile: Dict[str, Any], kind: str) -> Dict[str, Any]:
    for t in profile["topics"]:
        if t["kind"] == kind:
            return dict(t.get("facts", {}))
    return {}

def duplicate_visible_guard(new_text: str, last_text: str) -> bool:
    if not new_text or not last_text:
        return False
    a = new_text.strip().lower()
    b = last_text.strip().lower()
    if a == b:
        return True
    if len(a) >= 20 and a in b:
        return True
    if len(b) >= 20 and b in a:
        return True
    return False

# =======================
# STREAMLIT UI
# =======================
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="centered")
colA, colB = st.columns([1, 1])
with colA:
    st.title("Steckbrief")
    st.caption("Director decides • parallel search + vision • final text per item")
with colB:
    st.toggle("Debug", key="SHOW_DEBUG", help="Show raw JSON/text outputs for troubleshooting")

ensure_session()

# Harvest BG results into session
harvested = []
while True:
    try:
        harvested.append(BG_RESULTS.get_nowait())
    except queue.Empty:
        break

for res in harvested:
    topic_key = res.get("topic_key", "")
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
    # append result card if not duplicate image_url
    img_url = res.get("image", {}).get("image_url")
    dup = any((r.get("image", {}) or {}).get("image_url") == img_url and r.get("kind") == res.get("kind") for r in st.session_state.results)
    if not dup:
        st.session_state.results.append(res)
    # free inflight key
    st.session_state.inflight_keys.discard(topic_key)

# Topic badges
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

# Chat history (skip system)
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
        # snapshot before
        focus_before = current_topic()
        old_block = collect_topic_block(st.session_state.profile, focus_before)

        # director
        director_json = call_director(client, st.session_state.messages)
        st.session_state.last_director_json = director_json

        # topic focus hint
        topic_focus = director_json.get("topic_focus")
        if topic_focus in TOPIC_ORDER:
            set_current_topic(topic_focus)

        # facts
        facts_partial = director_json.get("facts_partial") or {}
        merge_facts(st.session_state.profile, facts_partial)

        # visible response w/ duplicate guard
        vis = (director_json.get("assistant_visible") or "").strip()
        if vis and not duplicate_visible_guard(vis, st.session_state.last_assistant_visible):
            st.session_state.messages.append({"role": "assistant", "content": vis})
            st.session_state.last_assistant_visible = vis

        # delta follow-up counting (only if current topic gained new facts)
        focus_now = current_topic()
        new_block = collect_topic_block(st.session_state.profile, focus_now)
        if focus_now == focus_before and facts_delta_count(old_block, new_block) > 0:
            inc_followup(focus_now)

        # actions (parallel & dedup)
        for act in (director_json.get("actions") or []):
            a_type = (act.get("type") or "").strip()
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
                start_media_task(client, a_type, "person", name, "", topic_key, {"facts": {"name": name}})
                # set topic status locally
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
                start_media_task(client, a_type, kind, title, author, topic_key, {"facts": {"title": title, "author_guess": author}})
                idx = ensure_topic(st.session_state.profile, kind, title)
                st.session_state.profile["topics"][idx]["status"] = "queued"

        # enforce hard follow-up budget and/or director hint
        if followups(current_topic()) >= MAX_FOLLOWUPS_PER_TOPIC:
            switch_to_next_topic(director_json.get("next_topic_hint"))
            # optional local nudge if director didn't provide a prompt for the next topic
            nxt = current_topic()
            nudge = {
                "podcast": "Thanks! Is there a podcast you’ve found inspiring lately?",
                "person": "Do you have a role model who has influenced your work?",
                "tool": "Is there a tool you rely on day-to-day?",
                "book": "Which book has helped you professionally?"
            }.get(nxt, "")
            if nudge and not duplicate_visible_guard(nudge, st.session_state.last_assistant_visible):
                st.session_state.messages.append({"role": "assistant", "content": nudge})
                st.session_state.last_assistant_visible = nudge

    else:
        st.session_state.messages.append({"role": "assistant",
                                          "content": "(Demo mode — set OPENAI_API_KEY in Streamlit Secrets.)"})
    st.rerun()

# Running tasks indicator
tasks_snapshot = bg_snapshot_tasks()
running = [t for t in tasks_snapshot.values() if t["status"] == "running"]
if running or st.session_state.inflight_keys:
    st.info("🔎 Background research running… you can keep chatting.")
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("🔄 Refresh results"):
            st.rerun()
    with c2:
        if st.session_state.SHOW_DEBUG:
            with st.expander("Tasks snapshot"):
                st.code(json.dumps(tasks_snapshot, indent=2))

# Error cards for failed tasks
failed = [t for t in tasks_snapshot.values() if t["status"] == "error"]
for ft in failed:
    st.warning(f"Background action failed for {ft.get('topic_key','?')}: {ft.get('error','Unknown error')}")

# Results
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
        for k in ["messages", "profile", "results", "last_assistant_visible", "topic_state", "inflight_keys",
                  "last_director_json", "last_raw_director_text"]:
            if k in st.session_state:
                del st.session_state[k]
        bg_clear_all()
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
