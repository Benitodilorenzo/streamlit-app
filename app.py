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
    {"name": "Book",    "followups": (1, 2), "focus": "books shaping strategy & Data/AI thinking"},
    {"name": "Podcast", "followups": (1, 2), "focus": "inspiring audio/video sources"},
    {"name": "Person",  "followups": (1, 2), "focus": "mentors or thought leaders"},
    {"name": "Tool",    "followups": (1, 2), "focus": "tools/methods for Data & AI Business Design"},
    {"name": "Idea",    "followups": (1, 2), "focus": "strategic concepts or AI approaches"},
]

GLOBAL_QUESTION_CAP = 12  # safety guard

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL      = os.getenv("OPENAI_CHAT_MODEL", "gpt-5")      # Director / Extractor / Vision
RESPONSES_MODEL = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-5") # web_search via Responses

SHOW_DEBUG = False   # True = Debug-Expander sichtbar

# =======================
# PROMPTS
# =======================
DIRECTOR_SYSTEM_PROMPT = (
    "You are a warm, curious interviewer crafting an 'Expert Card'.\n"
    "Domain: strategy, Data & AI Business Design; keep it focused but personable.\n"
    "Rules:\n"
    "- Topic order: Book → Podcast → Person → Tool → Idea.\n"
    "- Ask EXACTLY ONE short question per turn (never multiple). Avoid confirmations.\n"
    "- Vary angles: what/why/example/impact, within 1–2 follow-ups per topic (host controls budget).\n"
    "- Move on naturally once enough is gathered for the topic.\n"
    "- Stop when topics finished or global cap reached. Then thank the user succinctly.\n"
)

EXTRACTOR_SYSTEM_PROMPT = (
    "Extract one atomic value from the user's latest answer for the CURRENT TOPIC.\n"
    "Return EXACTLY one line (no extra text):\n"
    "Book → <title> | <author?>\n"
    "Podcast → <podcast/channel title>\n"
    "Person → <full name>\n"
    "Tool → <tool name>\n"
    "Idea → <short concept>\n"
    "If nothing extractable, return: —"
)

SEARCHER_SYSTEM_PROMPT = (
    "You are a researcher using a web search tool.\n"
    "Find high-quality candidate images that best represent the item.\n"
    "Heuristics:\n"
    "- Book: publisher/Amazon/Open Library/Wikipedia; avoid fan art.\n"
    "- Podcast: official channel art (YouTube/Spotify/Apple) or website.\n"
    "- Person: clear, respectful portrait (Wikipedia/official site preferred).\n"
    "- Tool: official logo/hero image.\n"
    "IMPORTANT:\n"
    "- Perform at most 3 focused searches. Stop early if you have 2–3 strong candidates.\n"
    "- Output 1–3 lines. Each line EXACTLY:\n"
    "IMAGE: <direct_image_url> | SOURCE: <source_page_url>\n"
    "No extra commentary."
)

VALIDATOR_SYSTEM_PROMPT = (
    "Pick the single best image among candidates for the target item.\n"
    "Criteria: clarity, correctness, professional look. No fan art.\n"
    "Respond in exactly TWO lines:\n"
    "BEST: <direct_image_url>\n"
    "REASON: <short reason>"
)

FINALIZER_SYSTEM_PROMPT = (
    "Compose a concise, warm, professional mini-profile ('Expert Card') from collected topics.\n"
    "Write 3–5 sentences summarizing the user with a positive, authentic tone (no flattery).\n"
    "Then provide a compact bullet per item (Book/Podcast/Person/Tool/Idea) with one short reason.\n"
    "Use the user's language if it’s clearly not English; otherwise English."
)

# =======================
# OPENAI
# =======================
def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing (Streamlit → Settings → Secrets).")
    return OpenAI(api_key=OPENAI_API_KEY)

# =======================
# STATE INIT
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
        for i, spec in enumerate(TOPICS_SPEC):
            lo, hi = spec["followups"]
            budget = random.randint(lo, hi)
            topics.append({
                "name": spec["name"],
                "status": "active" if i == 0 else "queued",  # first active
                "budget": budget,             # follow-ups allowed for this topic
                "answers": [],                # raw user strings
                "research": {},               # dict always (never None)
                "field": None,                # extracted atomic (title/author or name)
                "job_inflight": False         # background search guard
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
# OPENAI HELPERS
# =======================
def chat_stream_with_fallback(client: OpenAI, messages: List[Dict[str, Any]]) -> str:
    # try stream, fall back if org not verified
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
    except BadRequestError:
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
    # include up to 3 image URLs
    content: List[Dict[str, Any]] = [{"type": "text", "text": f"{VALIDATOR_SYSTEM_PROMPT}\nTarget: {item_desc}"}]
    added = 0
    for ln in candidate_lines:
        m = re.search(r"(?:IMAGE|CANDIDATE):\s*([^| \t]+)", ln, re.IGNORECASE)
        if not m:
            continue
        content.append({"type": "image_url", "image_url": {"url": m.group(1).strip()}})
        added += 1
        if added >= 3:
            break
    if added == 0:
        return None
    r = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": content}])
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
    elif topic_name in ("Podcast", "Tool", "Idea", "Person"):
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
            kind = job["kind"]
            title = job.get("title", "")
            author = job.get("author", "")
            name  = job.get("name", "")

            if kind == "book":
                item_desc = f'Book: "{title}" {author}'.strip()
                q = f"{SEARCHER_SYSTEM_PROMPT}\nQuery: Book cover: \"{title}\" {author}"
            elif kind == "podcast":
                item_desc = f'Podcast: "{title}"'
                q = f"{SEARCHER_SYSTEM_PROMPT}\nQuery: Podcast cover art: \"{title}\""
            elif kind == "person":
                item_desc = f'Person: \"{name}\"'
                q = f"{SEARCHER_SYSTEM_PROMPT}\nQuery: Portrait photo: \"{name}\""
            elif kind == "tool":
                item_desc = f'Tool: \"{title}\"'
                q = f"{SEARCHER_SYSTEM_PROMPT}\nQuery: Official logo or hero image: \"{title}\""
            else:
                item_desc = f'Idea: \"{title}\"'
                q = f"{SEARCHER_SYSTEM_PROMPT}\nQuery: Representative image/diagram: \"{title}\""

            txt = responses_web_search(client, q)
            # Accept IMAGE: ... or CANDIDATE: ... → normalize to IMAGE
            lines = []
            for ln in (txt or "").splitlines():
                s = ln.strip()
                if not s:
                    continue
                if not (s.upper().startswith("IMAGE:") or s.upper().startswith("CANDIDATE:")):
                    continue
                if ":" in s:
                    payload = s.split(":", 1)[1].strip()
                else:
                    payload = s
                # Try to pick first URL in the payload as image; also capture SOURCE url if present
                # Expected " <image> | SOURCE: <page>"
                image_url = ""
                page_url  = ""
                # simple split on '|' parts
                parts = [p.strip() for p in payload.split("|")]
                if parts:
                    # first token may already be a URL
                    first = parts[0]
                    # remove optional "SOURCE:" from first if misordered
                    if first.lower().startswith("source:"):
                        # then no image, skip
                        continue
                    image_url = first
                for p in parts[1:]:
                    if p.lower().startswith("source:"):
                        page_url = p.split(":", 1)[1].strip()
                if image_url:
                    lines.append(f"IMAGE: {image_url} | SOURCE: {page_url}")

            best = vision_pick_best(client, item_desc, lines) if lines else None

            out_q.put({"ok": True, "job_id": job["job_id"], "kind": kind,
                       "candidates": lines, "best": best})
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
                for t in st.session_state.profile["topics"]:
                    if t["name"].lower() == kind:
                        t["research"] = {
                            "candidates": res.get("candidates") or [],
                            "best": res.get("best") or ""
                        }
                        t["job_inflight"] = False
                        changed = True
                        break
            else:
                st.session_state.debug_log.append(f"[BG ERROR] {res.get('error')}")
        finally:
            st.session_state.bg_result_queue.task_done()
    return changed

# =======================
# ENQUEUE SEARCH
# =======================
def enqueue_search_if_ready(client: OpenAI, topic: Dict[str, Any], user_text: str):
    research = topic.get("research") or {}
    if topic.get("job_inflight") or research.get("best") or research.get("candidates"):
        return

    # extract atomic field
    recent_msgs = [m for m in st.session_state.history if m["role"] != "system"][-6:]
    recent = " | ".join(m["content"] for m in recent_msgs)[-600:]
    fields = extractor_pull_fields(client, topic["name"], user_text, recent)
    if not fields:
        return
    topic["field"] = fields

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
    topic["job_inflight"] = True
    if SHOW_DEBUG:
        st.session_state.debug_log.append(f"[ENQ] {job}")

# =======================
# FINALIZER
# =======================
def build_final_text(client: OpenAI, profile: Dict[str, Any]) -> str:
    bullets = []
    for t in profile["topics"]:
        name = t["name"]
        atom = (t.get("field") or {}).get("title") or (t.get("field") or {}).get("name") or ""
        why = t["answers"][-1] if t["answers"] else ""
        bullets.append(f"- {name}: {atom} — {why}".strip())
    facts = "Interview facts:\n" + "\n".join(bullets)
    return chat_once(client, FINALIZER_SYSTEM_PROMPT, facts)

# =======================
# UI
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

def enforce_single_question(text: str) -> str:
    """Director may try multiple lines; keep only first question/sentence."""
    if not text:
        return text
    # split by question mark or newline; keep first non-empty chunk
    first = re.split(r'\?\s+|\n+', text.strip(), maxsplit=1)[0]
    # if original had a '?', add it back to keep it a question
    if "?" in text:
        return first.strip() + "?"
    return first.strip()

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

# Client + worker
agent_ready = True
try:
    client = get_client()
    # Start background worker once
    th = st.session_state.bg_threads.get("main")
    if not th or not th.is_alive():
        ensure_worker(client)
except Exception as e:
    agent_ready = False
    st.warning(f"OpenAI not configured: {e}")

# Drain finished background results (causes UI refresh when something arrived)
if drain_bg_results():
    st.rerun()

# Timeline
render_timeline(st.session_state.profile)

# Show chat so far (skip system)
for m in st.session_state.history:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
user_text = st.chat_input("Type your answer…")
if user_text:
    st.session_state.history.append({"role": "user", "content": user_text})

    topic = current_topic(st.session_state.profile)
    topic["answers"].append(user_text)

    if agent_ready:
        enqueue_search_if_ready(client, topic, user_text)

    # Ask Director only if we still have follow-up budget for this topic
    if agent_ready and topic["budget"] > 0 and can_ask_more(st.session_state.profile):
        # Ask for the next (single) question
        reply = chat_stream_with_fallback(client, st.session_state.history[-12:])
        reply = enforce_single_question(reply)
        st.session_state.profile["bot_questions"] += 1
        topic["budget"] -= 1                  # <-- budget sink occurs when bot asks
        st.session_state.history.append({"role": "assistant", "content": reply})

        # If budget is now 0 → advance to next topic
        if topic["budget"] <= 0:
            topic["status"] = "done"
            advance_topic(st.session_state.profile)

    else:
        # No more questions for this topic → advance (if not already done)
        if topic["status"] == "active":
            topic["status"] = "done"
            advance_topic(st.session_state.profile)

        # If there are remaining topics, prompt first question for the new topic
        if agent_ready and not all_topics_done(st.session_state.profile) and can_ask_more(st.session_state.profile):
            reply = chat_stream_with_fallback(client, st.session_state.history[-12:])
            reply = enforce_single_question(reply)
            st.session_state.profile["bot_questions"] += 1
            next_topic = current_topic(st.session_state.profile)
            if next_topic["budget"] > 0:
                next_topic["budget"] -= 1
            st.session_state.history.append({"role": "assistant", "content": reply})

    # End condition: either all topics done or cap reached
    if all_topics_done(st.session_state.profile) or not can_ask_more(st.session_state.profile):
        if not st.session_state.finalized:
            st.session_state.history.append({
                "role": "assistant",
                "content": "Thanks — I’ve got a solid picture now. I’ll assemble your expert card in the background."
            })
            st.session_state.finalized = True

    st.rerun()

# Final assembly (once)
if st.session_state.finalized and agent_ready and not st.session_state.final_text:
    # small grace so background search can populate some images
    time.sleep(0.2)
    st.session_state.final_text = build_final_text(client, st.session_state.profile)

# Output table + export
if st.session_state.finalized:
    st.subheader("Expert Card")
    rows = []
    for t in st.session_state.profile["topics"]:
        name = t["name"]
        atom = (t.get("field") or {}).get("title") or (t.get("field") or {}).get("name") or ""
        why  = t["answers"][-1] if t["answers"] else ""
        img  = (t.get("research") or {}).get("best") or ""
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
            header = f"**{name}**"
            if atom:
                header += f" — {atom}"
            st.markdown(header)
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
    st.download_button(
        "⬇️ Download JSON",
        data=json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="expert_card.json",
        mime="application/json",
    )
