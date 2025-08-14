import os, json, re, time, random, threading, queue, uuid
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI

# =======================
# CONFIG
# =======================
APP_TITLE = "📝 Expert Card Creator"

# Topic spectrum (strategy / Data & AI Business Design focus)
# followups = (min, max) — randomized per session to keep it natural
TOPICS_SPEC = [
    {"name": "Book",    "followups": (1, 3), "focus": "books shaping their strategy & Data/AI thinking"},
    {"name": "Podcast", "followups": (1, 3), "focus": "audio/video sources inspiring professional growth"},
    {"name": "Person",  "followups": (1, 2), "focus": "mentors or thought leaders they admire"},
    {"name": "Tool",    "followups": (1, 2), "focus": "tools, frameworks, methods used in Data & AI Business Design"},
    {"name": "Idea",    "followups": (1, 2), "focus": "strategic concepts, data principles, or AI approaches"},
]

GLOBAL_QUESTION_CAP = 12  # safety: stop after N bot questions to avoid endless chat

# Models (override via Streamlit Secrets or env as needed)
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL      = os.getenv("OPENAI_CHAT_MODEL", "gpt-5")  # Director / Extractor / Vision
RESPONSES_MODEL = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-5")  # for web_search tool (Responses API)

# UI flags
SHOW_DEBUG = False  # set True during development

# =======================
# SYSTEM PROMPTS
# =======================
DIRECTOR_SYSTEM_PROMPT = (
    "You are a warm, curious interviewer crafting an 'Expert Card' about the user.\n"
    "Domain focus: strategy, Data & AI Business Design, and Data & AI in general, "
    "but you may explore tangentially if it helps capture personality.\n\n"
    "Conversation rules:\n"
    "- Follow the planned sequence of topics from the host app (Book → Podcast → Person → Tool → Idea).\n"
    "- For each topic, ask between 1 and 3 short, meaningful follow-up questions (the host sets a per-topic budget).\n"
    "- Avoid redundant confirmations; never mention background tools or searches.\n"
    "- Keep the tone concise, engaging, and professional.\n"
    "- When the current topic seems sufficiently covered, move on to the next topic naturally.\n"
    "- When all topics are covered (or a global question cap is reached), end with a brief thanks.\n"
)

EXTRACTOR_SYSTEM_PROMPT = (
    "You extract atomic fields from the user's latest answer for the CURRENT TOPIC.\n"
    "Return exactly ONE line. No extra commentary.\n\n"
    "Formats:\n"
    "- Book → <title> | <author?>\n"
    "- Podcast → <podcast or channel title>\n"
    "- Person → <full name>\n"
    "- Tool → <tool name>\n"
    "- Idea → <short concept>\n"
    "If you cannot extract a clean value, return just: —"
)

SEARCHER_SYSTEM_PROMPT = (
    "You are a researcher using a web search tool. Find up to 3 high-quality candidate images that best represent the item.\n"
    "Heuristics:\n"
    "- Book: official/publisher, Amazon, Open Library, Wikipedia; avoid fan art, memes.\n"
    "- Podcast: official channel art (YouTube/Spotify/Apple) or website brand image.\n"
    "- Person: a clear, respectful portrait (Wikipedia or official site preferred).\n"
    "- Tool: official product logo or hero image.\n\n"
    "Output format: For each candidate, exactly one line:\n"
    "IMAGE: <direct_image_url> | SOURCE: <source_page_url>\n"
    "No extra commentary."
)

VALIDATOR_SYSTEM_PROMPT = (
    "Pick the single best image among candidates for the target item. Consider clarity, correctness, and professional look.\n"
    "Respond in exactly TWO lines:\n"
    "BEST: <direct_image_url>\n"
    "REASON: <short reason>"
)

FINALIZER_SYSTEM_PROMPT = (
    "You compose a concise, warm, professional mini-profile ('Expert Card') from collected topics. "
    "Write 3–5 sentences summarizing the user, with a positive, authentic tone (no flattery). "
    "Then provide a compact bullet per item (Book/Podcast/Person/Tool/Idea) with a one-liner why it matters.\n"
    "Keep it crisp. Use the user's language if it's clearly not English; otherwise default to English."
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
        # Create topics with randomized followup budgets
        topics = []
        for spec in TOPICS_SPEC:
            lo, hi = spec["followups"]
            budget = random.randint(lo, hi)
            topics.append({
                "name": spec["name"],
                "status": "queued",     # queued | active | done
                "budget": budget,       # remaining followups (post first answer)
                "answers": [],          # user answers (strings)
                "research": None,       # {"candidates":[...], "best": url} when ready
                "field": None           # extracted atomic value (title/name/idea)
            })
        topics[0]["status"] = "active"
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
    for j in range(ix+1, len(topics)):
        if topics[j]["status"] == "queued":
            topics[j]["status"] = "active"
            profile["current_topic_index"] = j
            return
    profile["current_topic_index"] = ix  # no more, stays at last

def all_topics_done(profile: Dict[str, Any]) -> bool:
    return all(t["status"] == "done" for t in profile["topics"])

# =======================
# OPENAI CALL HELPERS
# =======================
def chat_stream(client: OpenAI, messages: List[Dict[str, Any]]) -> str:
    """Director streaming reply (Chat Completions). No temperature/top_p."""
    out = []
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
    """
    Responses API + built-in web_search tool.
    We ask it to output one line per candidate:
    IMAGE: <direct_image_url> | SOURCE: <page_url>
    """
    r = client.responses.create(
        model=RESPONSES_MODEL,
        tools=[{"type": "web_search"}],
        input=[{"role": "user", "content": [{"type": "input_text", "text": query_text}]}],
    )
    # Prefer unified .output_text if present
    txt = getattr(r, "output_text", None)
    if txt is None and getattr(r, "output", None):
        try:
            txt = r.output[0]["content"][0]["text"]
        except Exception:
            txt = ""
    return txt or ""

def vision_pick_best(client: OpenAI, item_desc: str, lines: List[str]) -> Optional[str]:
    """
    Vision validator: provide up to 3 images and get 'BEST: <url>' back.
    """
    content: List[Dict[str, Any]] = [{"type": "text", "text": f"{VALIDATOR_SYSTEM_PROMPT}\nTarget: {item_desc}"}]
    added = 0
    for ln in lines:
        m = re.search(r"IMAGE:\s*(\S+)", ln)
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
    """
    LLM-based field extractor; returns dict with a single atomic field per topic.
    """
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
            title = job.get("title")    # book/podcast/tool/idea
            author = job.get("author", "")
            name  = job.get("name")     # person
            # Build query text
            if kind == "book":
                query = f"{SEARCHER_SYSTEM_PROMPT}\nQuery: Book cover: \"{title}\" {author}".strip()
                item_desc = f'Book: "{title}" {author}'.strip()
            elif kind == "podcast":
                query = f"{SEARCHER_SYSTEM_PROMPT}\nQuery: Podcast cover art: \"{title}\""
                item_desc = f'Podcast: "{title}"'
            elif kind == "person":
                query = f"{SEARCHER_SYSTEM_PROMPT}\nQuery: Portrait photo: \"{name}\""
                item_desc = f'Person: "{name}"'
            elif kind == "tool":
                query = f"{SEARCHER_SYSTEM_PROMPT}\nQuery: Official logo or hero image: \"{title}\""
                item_desc = f'Tool: "{title}"'
            else:  # idea
                query = f"{SEARCHER_SYSTEM_PROMPT}\nQuery: Concept visual/diagram or representative image: \"{title}\""
                item_desc = f'Idea: "{title}"'

            txt = responses_web_search(client, query)
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip().lower().startswith("image:")]
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
    if "bg_threads" not in st.session_state:
        st.session_state.bg_threads = {}
    th = st.session_state.bg_threads.get("main")
    if not th or not th.is_alive():
        t = threading.Thread(target=bg_worker, args=(client, st.session_state.bg_task_queue, st.session_state.bg_result_queue), daemon=True)
        st.session_state.bg_threads["main"] = t
        t.start()

def drain_bg_results():
    """Main thread: consume results and write into session_state safely."""
    changed = False
    while True:
        try:
            res = st.session_state.bg_result_queue.get_nowait()
        except queue.Empty:
            break
        if res.get("ok"):
            kind = res["kind"]
            for t in st.session_state.profile["topics"]:
                if t["name"].lower() == kind:
                    t["research"] = {"candidates": res["candidates"], "best": res["best"]}
                    changed = True
                    break
        else:
            st.session_state.debug_log.append(f"[BG ERROR] {res.get('error')}")
        st.session_state.bg_result_queue.task_done()
    return changed

# =======================
# SEARCH TRIGGER
# =======================
def enqueue_search_if_ready(client: OpenAI, topic: Dict[str, Any], user_text: str):
    """
    Let the extractor decide if we have the atomic field; if yes, enqueue search job once.
    """
    # Skip if already have field or research
    if topic.get("field") or topic.get("research"):
        return
    # Use recent context (last 6 messages excluding system)
    recent = []
    for m in reversed([m for m in st.session_state.history if m["role"] != "system"][-6:]):
        recent.append(m["content"])
    short_ctx = " | ".join(recent[::-1])[:600]

    fields = extractor_pull_fields(client, topic["name"], user_text, short_ctx)
    if not fields:
        return

    topic["field"] = fields  # store structured atomic field

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
    else:  # idea
        job.update({"kind": "idea", "title": fields.get("name","")})

    st.session_state.bg_task_queue.put(job)
    if SHOW_DEBUG:
        st.session_state.debug_log.append(f"[ENQ] {job}")

# =======================
# FINALIZER
# =======================
def build_final_text(client: OpenAI, profile: Dict[str, Any]) -> str:
    # Collect concise facts per topic
    bullets = []
    for t in profile["topics"]:
        name = t["name"]
        # prefer the last answer for the one-liner seed
        why = t["answers"][-1] if t["answers"] else ""
        field = t.get("field") or {}
        atom = field.get("title") or field.get("name") or ""
        img = (t.get("research") or {}).get("best") or ""
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
            if t.get("research", {}).get("best"):
                st.image(t["research"]["best"], caption="Selected", use_column_width=True)

def can_ask_more(profile: Dict[str, Any]) -> bool:
    return profile["bot_questions"] < GLOBAL_QUESTION_CAP

# =======================
# STREAMLIT APP
# =======================
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="centered")
st.title(APP_TITLE)
st.caption("Director (GPT-5 Chat) → Extraction → Web Search (Responses) → Vision validation → Final table")

init_state()

if SHOW_DEBUG:
    with st.expander("🔧 Debug", expanded=False):
        st.json(st.session_state.debug_log)

# Render timeline
render_timeline(st.session_state.profile)

# Render chat so far (skip system)
for m in st.session_state.history:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Client
agent_ready = True
try:
    client = get_client()
    ensure_worker(client)
except Exception as e:
    agent_ready = False
    st.warning(f"OpenAI not configured: {e}")

# Drain any finished background results
if drain_bg_results():
    st.rerun()

# User input
user_text = st.chat_input("Type your answer…")
if user_text:
    st.session_state.history.append({"role": "user", "content": user_text})

    # Bookkeeping per topic
    topic = current_topic(st.session_state.profile)
    topic["answers"].append(user_text)

    # Trigger search if ready (Extractor decides)
    if agent_ready:
        enqueue_search_if_ready(client, topic, user_text)

    # Decide to move on or keep follow-ups
    if topic["budget"] > 0:
        topic["budget"] -= 1
    else:
        # move to next topic
        topic["status"] = "done"
        advance_topic(st.session_state.profile)

    # Director replies if allowed
    reply_text = "Thanks — I’ll assemble your expert card."
    if agent_ready and not all_topics_done(st.session_state.profile) and can_ask_more(st.session_state.profile):
        # ask next question with recent context only
        reply_text = chat_stream(client, st.session_state.history[-12:])
        st.session_state.profile["bot_questions"] += 1

    st.session_state.history.append({"role": "assistant", "content": reply_text})

    # End when cap reached or all topics done
    if all_topics_done(st.session_state.profile) or not can_ask_more(st.session_state.profile):
        if not st.session_state.finalized:
            st.session_state.history.append({"role": "assistant",
                                             "content": "Thanks, I’ve got a solid picture now. I’ll assemble your expert card in the background."})
            st.session_state.finalized = True
    st.rerun()

# Final assembly (once)
if st.session_state.finalized and agent_ready and not st.session_state.final_text:
    # give background worker time to complete any remaining searches
    time.sleep(0.2)
    final_text = build_final_text(client, st.session_state.profile)
    st.session_state.final_text = final_text

# Output table + download
if st.session_state.finalized:
    st.subheader("Expert Card (compact)")
    # Rows
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
