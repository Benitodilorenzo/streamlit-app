# app.py — Expert Card Creator (fixed)

import os, json, re, time, random, threading, queue, uuid
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI
from openai._exceptions import BadRequestError

# =======================
# CONFIG
# =======================
APP_TITLE = "📝 Expert Card Creator"

TOPICS_SPEC = [
    {"name": "Book",    "followups": (1, 3), "focus": "books shaping strategy & Data/AI thinking"},
    {"name": "Podcast", "followups": (1, 3), "focus": "inspiring audio/video sources"},
    {"name": "Person",  "followups": (1, 2), "focus": "mentors or thought leaders"},
    {"name": "Tool",    "followups": (1, 2), "focus": "tools/methods for Data & AI Business Design"},
    {"name": "Idea",    "followups": (1, 2), "focus": "strategic concepts or AI approaches"},
]

GLOBAL_QUESTION_CAP = 12  # prevent endless interviews

# Models (override via Secrets/env)
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL      = os.getenv("OPENAI_CHAT_MODEL", "gpt-5")  # director/extractor/vision
RESPONSES_MODEL = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-5")  # for web_search via Responses API

# UI flags
SHOW_DEBUG = False   # True = show internal logs box

# =======================
# SYSTEM PROMPTS
# =======================
DIRECTOR_SYSTEM_PROMPT = (
    "You are a warm, curious interviewer crafting an 'Expert Card' about the user.\n"
    "Domain focus: strategy, Data & AI Business Design, and Data & AI in general, with room for personality.\n\n"
    "Rules:\n"
    "- Follow host order: Book → Podcast → Person → Tool → Idea.\n"
    "- For each topic, ask 1–3 short follow-ups (budget set by host). Keep it focused and varied.\n"
    "- Avoid redundant confirmations and never mention background tools.\n"
    "- Move on naturally when enough for the topic is gathered.\n"
    "- Stop when all topics covered or a global cap is reached; then thank the user.\n"
)

EXTRACTOR_SYSTEM_PROMPT = (
    "Extract a single atomic value from the user's latest answer for the CURRENT TOPIC.\n"
    "Return EXACTLY ONE line. No extra text.\n"
    "Formats:\n"
    "- Book → <title> | <author?>\n"
    "- Podcast → <podcast/channel title>\n"
    "- Person → <full name>\n"
    "- Tool → <tool name>\n"
    "- Idea → <short concept>\n"
    "If nothing extractable, return: —"
)

SEARCHER_SYSTEM_PROMPT = (
    "You are a web image search assistant using a web_search tool. Find up to 3 high-quality candidate images.\n"
    "Heuristics:\n"
    "- Book: publisher/Amazon/Open Library/Wikipedia; avoid fan art.\n"
    "- Podcast: official channel art (YouTube/Spotify/Apple) or site logo.\n"
    "- Person: clear, respectful portrait (Wikipedia or official site).\n"
    "- Tool: official logo/hero image.\n"
    "Output format: Each candidate on its own line exactly like:\n"
    "CANDIDATE: <direct_image_url> | <page_url> | <source>\n"
    "No extra commentary."
)

VALIDATOR_SYSTEM_PROMPT = (
    "Choose the single best image among candidates for the target item. Consider clarity, correctness, and professional look.\n"
    "Respond in exactly TWO lines:\n"
    "BEST: <direct_image_url>\n"
    "REASON: <short reason>"
)

FINALIZER_SYSTEM_PROMPT = (
    "Compose a concise, warm, professional mini-profile ('Expert Card') from collected topics.\n"
    "Write 3–5 sentences summarizing the user with a positive, authentic tone (no flattery).\n"
    "Then provide a compact bullet per item (Book/Podcast/Person/Tool/Idea) with one short reason why it matters.\n"
    "Use the user's language if it's clearly not English; else English."
)

# =======================
# OPENAI CLIENT
# =======================
def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing (Streamlit → Settings → Secrets).")
    return OpenAI(api_key=OPENAI_API_KEY)

# =======================
# STATE
# =======================
def init_state():
    if "history" not in st.session_state:
        st.session_state.history: List[Dict[str, Any]] = [
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "assistant", "content":
                "Hi! I'll help you craft a tiny expert card. "
                "First, which book has helped you professionally? Please share the title (author optional)."}
        ]
    if "profile" not in st.session_state:
        topics = []
        for spec in TOPICS_SPEC:
            lo, hi = spec["followups"]
            budget = random.randint(lo, hi)
            topics.append({
                "name": spec["name"],
                "status": "active" if not topics else "queued",  # first is active
                "budget": budget,
                "answers": [],
                "research": {},        # NEVER None
                "field": None          # extracted atomic value dict
            })
        st.session_state.profile = {
            "topics": topics,
            "current_topic_index": 0,
            "bot_questions": 0
        }
    if "bg_task_queue" not in st.session_state:
        st.session_state.bg_task_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    if "bg_result_queue" not in st.session_state:
        st.session_state.bg_result_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    if "bg_threads" not in st.session_state:
        st.session_state.bg_threads: Dict[str, threading.Thread] = {}
    if "finalized" not in st.session_state:
        st.session_state.finalized = False
    if "final_text" not in st.session_state:
        st.session_state.final_text = ""
    if "debug_log" not in st.session_state:
        st.session_state.debug_log = []

def current_topic(profile: Dict[str, Any]) -> Dict[str, Any]:
    return profile["topics"][profile["current_topic_index"]]

def advance_topic(profile: Dict[str, Any]):
    ix = profile["current_topic_index"]
    topics = profile["topics"]
    topics[ix]["status"] = "done"
    for j in range(ix + 1, len(topics)):
        if topics[j]["status"] == "queued":
            topics[j]["status"] = "active"
            profile["current_topic_index"] = j
            return

def all_topics_done(profile: Dict[str, Any]) -> bool:
    return all(t["status"] == "done" for t in profile["topics"])

def can_ask_more(profile: Dict[str, Any]) -> bool:
    return profile["bot_questions"] < GLOBAL_QUESTION_CAP

# =======================
# OPENAI CALL HELPERS
# =======================
def chat_stream_with_fallback(client: OpenAI, messages: List[Dict[str, Any]]) -> str:
    """Try streaming; if 400 unsupported, fall back to one-shot."""
    out = []
    try:
        with st.chat_message("assistant"):
            ph = st.empty()
            stream = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                if delta:
                    out.append(delta)
                    ph.markdown("".join(out))
        return "".join(out).strip()
    except BadRequestError as e:
        # fall back to non-stream
        r = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
        text = r.choices[0].message.content or ""
        with st.chat_message("assistant"):
            st.markdown(text)
        return text

def chat_once(client: OpenAI, system_prompt: str, user_text: str) -> str:
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    return r.choices[0].message.content or ""

def responses_web_search(client: OpenAI, query_text: str) -> str:
    r = client.responses.create(
        model=RESPONSES_MODEL,
        tools=[{"type": "web_search"}],
        input=[{"role": "user", "content": [{"type": "input_text", "text": query_text}]}],
    )
    txt = getattr(r, "output_text", None)
    if txt is None and getattr(r, "output", None):
        try:
            txt = r.output[0]["content"][0]["text"]
        except Exception:
            txt = ""
    return txt or ""

def vision_pick_best(client: OpenAI, item_desc: str, candidate_lines: List[str]) -> Optional[str]:
    # Feed up to 3 images to validator
    content: List[Dict[str, Any]] = [{"type": "text", "text": f"{VALIDATOR_SYSTEM_PROMPT}\nTarget: {item_desc}"}]
    added = 0
    for ln in candidate_lines:
        m = re.search(r"CANDIDATE:\s*([^| \t]+)", ln, re.IGNORECASE)
        if not m:
            continue
        content.append({"type": "image_url", "image_url": {"url": m.group(1).strip()}})
        added += 1
        if added >= 3:
            break
    if added == 0:
        return None
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": content}],
    )
    txt = r.choices[0].message.content or ""
    m = re.search(r"BEST:\s*(\S+)", txt)
    return m.group(1).strip() if m else None

# =======================
# EXTRACTOR
# =======================
def extractor_pull_fields(client: OpenAI, topic_name: str, user_text: str, recent_context: str = "") -> Optional[Dict[str, str]]:
    prompt = f"CURRENT TOPIC: {topic_name}\nContext: {recent_context}\nUser said: {user_text}\nExtract:"
    line = chat_once(client, EXTRACTOR_SYSTEM_PROMPT, prompt).strip()
    if not line or line == "—":
        return None
    if topic_name == "Book":
        parts = [p.strip() for p in line.split("|", 1)]
        title = parts[0] if parts else ""
        author = parts[1] if len(parts) > 1 else ""
        if not title:
            return None
        return {"title": title, "author": author}
    elif topic_name in ("Podcast", "Tool", "Idea"):
        return {"name": line}
    elif topic_name == "Person":
        return {"name": line}
    return None

# =======================
# BACKGROUND WORKER
# =======================
def bg_worker(client: OpenAI, in_q: "queue.Queue[Dict[str, Any]]", out_q: "queue.Queue[Dict[str, Any]]"):
    while True:
        job = in_q.get()
        if job is None:
            in_q.task_done()
            break
        try:
            kind = job["kind"]  # "book"|"podcast"|"person"|"tool"|"idea"
            title = job.get("title", "")
            author = job.get("author", "")
            name  = job.get("name", "")

            # Build query
            if kind == "book":
                item_desc = f'Book: "{title}" {author}'.strip()
                query = f"{SEARCHER_SYSTEM_PROMPT}\nQuery: Book cover: \"{title}\" {author}"
            elif kind == "podcast":
                item_desc = f'Podcast: "{title}"'
                query = f"{SEARCHER_SYSTEM_PROMPT}\nQuery: Podcast cover art: \"{title}\""
            elif kind == "person":
                item_desc = f'Person: "{name}"'
                query = f"{SEARCHER_SYSTEM_PROMPT}\nQuery: Portrait photo: \"{name}\""
            elif kind == "tool":
                item_desc = f'Tool: "{title}"'
                query = f"{SEARCHER_SYSTEM_PROMPT}\nQuery: Official logo or hero image: \"{title}\""
            else:
                item_desc = f'Idea: "{title}"'
                query = f"{SEARCHER_SYSTEM_PROMPT}\nQuery: Representative image/diagram: \"{title}\""

            txt = responses_web_search(client, query)
            # Normalize to our single format
            lines = []
            for ln in (txt or "").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                if not ln.upper().startswith("CANDIDATE:"):
                    continue
                # ensure we have at least the image url
                payload = ln.split(":", 1)[1].strip() if ":" in ln else ""
                # expect "<image> | <page> | <source>" but tolerate missing parts
                parts = [p.strip() for p in payload.split("|")]
                image_url = parts[0] if len(parts) >= 1 else ""
                page_url  = parts[1] if len(parts) >= 2 else ""
                source    = parts[2] if len(parts) >= 3 else ""
                if image_url:
                    lines.append(f"CANDIDATE: {image_url} | {page_url} | {source}")

            best = vision_pick_best(client, item_desc, lines) if lines else None

            out_q.put({
                "ok": True,
                "job_id": job["job_id"],
                "kind": kind,
                "candidates": lines,
                "best": best
            })
        except Exception as e:
            out_q.put({"ok": False, "job_id": job.get("job_id",""), "error": str(e)})
        finally:
            in_q.task_done()

def ensure_worker(client: OpenAI):
    th = st.session_state.bg_threads.get("main")
    if not th or not th.is_alive():
        t = threading.Thread(target=bg_worker,
                             args=(client, st.session_state.bg_task_queue, st.session_state.bg_result_queue),
                             daemon=True)
        st.session_state.bg_threads["main"] = t
        t.start()

def drain_bg_results():
    changed = False
    while True:
        try:
            res = st.session_state.bg_result_queue.get_nowait()
        except queue.Empty:
            break
        try:
            if res.get("ok"):
                kind = res["kind"]
                # write back safely (research dict, never None)
                for t in st.session_state.profile["topics"]:
                    if t["name"].lower() == kind:
                        t["research"] = {
                            "candidates": res.get("candidates") or [],
                            "best": res.get("best") or ""
                        }
                        changed = True
                        break
            else:
                st.session_state.debug_log.append(f"[BG ERROR] {res.get('error')}")
        finally:
            st.session_state.bg_result_queue.task_done()
    return changed

# =======================
# SEARCH TRIGGER
# =======================
def enqueue_search_if_ready(client: OpenAI, topic: Dict[str, Any], user_text: str):
    # Skip if field already extracted or research already present
    research = topic.get("research") or {}
    if topic.get("field") or research.get("best") or research.get("candidates"):
        return

    # Light context for extractor
    recent = []
    for m in reversed([m for m in st.session_state.history if m["role"] != "system"][-6:]):
        recent.append(m["content"])
    short_ctx = " | ".join(recent[::-1])[:600]

    fields = extractor_pull_fields(client, topic["name"], user_text, short_ctx)
    if not fields:
        return

    topic["field"] = fields  # store atomic value

    job = {"job_id": str(uuid.uuid4())[:8]}
    nm = topic["name"].lower()
    if nm == "book":
        job.update({"kind": "book", "title": fields.get("title",""), "author": fields.get("author","")})
    elif nm == "podcast":
        job.update({"kind": "podcast", "title": fields.get("name","")})
    elif nm == "person":
        job.update({"kind": "person", "name": fields.get("name","")})
    elif nm == "tool":
        job.update({"kind": "tool", "title": fields.get("name","")})
    else:
        job.update({"kind": "idea", "title": fields.get("name","")})

    st.session_state.bg_task_queue.put(job)
    if SHOW_DEBUG:
        st.session_state.debug_log.append(f"[ENQ] {job}")

# =======================
# FINALIZER
# =======================
def build_final_text(client: OpenAI, profile: Dict[str, Any]) -> str:
    bullets = []
    for t in profile["topics"]:
        name = t["name"]
        why = t["answers"][-1] if t["answers"] else ""
        field = t.get("field") or {}
        atom = field.get("title") or field.get("name") or ""
        bullets.append(f"- {name}: {atom} — {why}".strip())
    facts = "Interview facts:\n" + "\n".join(bullets)
    return chat_once(client, FINALIZER_SYSTEM_PROMPT, facts)

# =======================
# UI HELPERS
# =======================
def render_timeline(profile: Dict[str, Any]):
    cols = st.columns(len(profile["topics"]))
    for i, t in enumerate(profile["topics"]):
        with cols[i]:
            cur = " (current)" if i == profile["current_topic_index"] and t["status"] == "active" else ""
            st.markdown(f"**{t['name']}**{cur}")
            st.caption(f"status: {t['status']} • followups left: {t['budget']}")
            best_url = (t.get("research") or {}).get("best")
            if best_url:
                st.image(best_url, caption="Selected", use_column_width=True)

# =======================
# APP
# =======================
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="centered")
st.title(APP_TITLE)
st.caption("Director (GPT-5 Chat) → Extraction → Web Search (Responses) → Vision → Final table")

init_state()

if SHOW_DEBUG:
    with st.expander("🔧 Debug", expanded=False):
        st.json(st.session_state.debug_log)

render_timeline(st.session_state.profile)

# Render chat history (skip system)
for m in st.session_state.history:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# OpenAI client
agent_ready = True
try:
    client = get_client()
    ensure_worker(client)
except Exception as e:
    agent_ready = False
    st.warning(f"OpenAI not configured: {e}")

# Consume finished background results
if drain_bg_results():
    st.rerun()

# Chat input
user_text = st.chat_input("Type your answer…")
if user_text:
    st.session_state.history.append({"role": "user", "content": user_text})

    topic = current_topic(st.session_state.profile)
    topic["answers"].append(user_text)

    if agent_ready:
        enqueue_search_if_ready(client, topic, user_text)

    # budget logic → move on when exhausted
    if topic["budget"] > 0:
        topic["budget"] -= 1
    else:
        topic["status"] = "done"
        advance_topic(st.session_state.profile)

    # Director reply (with stream fallback)
    reply_text = "Thanks — I’ll assemble your expert card."
    if agent_ready and not all_topics_done(st.session_state.profile) and can_ask_more(st.session_state.profile):
        reply_text = chat_stream_with_fallback(client, st.session_state.history[-12:])
        st.session_state.profile["bot_questions"] += 1

    st.session_state.history.append({"role": "assistant", "content": reply_text})

    # End condition
    if all_topics_done(st.session_state.profile) or not can_ask_more(st.session_state.profile):
        if not st.session_state.finalized:
            st.session_state.history.append({
                "role": "assistant",
                "content": "Thanks, I’ve got a solid picture now. I’ll assemble your expert card in the background."
            })
            st.session_state.finalized = True
    st.rerun()

# Final assembly (once)
if st.session_state.finalized and agent_ready and not st.session_state.final_text:
    time.sleep(0.2)  # brief grace for background
    st.session_state.final_text = build_final_text(client, st.session_state.profile)

# Output table + export
if st.session_state.finalized:
    st.subheader("Expert Card (compact)")
    rows = []
    for t in st.session_state.profile["topics"]:
        name = t["name"]
        atom = (t.get("field") or {}).get("title") or (t.get("field") or {}).get("name") or ""
        why = t["answers"][-1] if t["answers"] else ""
        img = (t.get("research") or {}).get("best") or ""
        rows.append((name, atom, why, img))

    for name, atom, why, img in rows:
        c1, c2 = st.columns([1, 2])
        with c1:
            if img:
                st.image(img, caption=name, use_column_width=True)
            else:
                st.markdown(f"**{name}**")
                st.caption("No image yet.")
        with c2:
            st.markdown(f"**{name}** — {atom if atom else ''}")
            st.write(why if why else "_—_")

    if st.session_state.final_text:
        st.markdown("---")
        st.markdown("**Summary**")
        st.write(st.session_state.final_text)

    export = {
        "profile": st.session_state.profile,
        "history": [m for m in st.session_state.history if m["role"] != "system"],
        "final_text": st.session_state.final_text,
    }
    st.download_button("⬇️ Download JSON",
                       data=json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="expert_card.json",
                       mime="application/json")
