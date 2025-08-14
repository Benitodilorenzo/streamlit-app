import os, time, re, threading, queue, uuid, urllib.parse
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI

# =======================
# CONFIG
# =======================
APP_TITLE = "📝 Expert Card Creator"

# Topics (breit, aber mit klarer Guideline). budget = max Follow-ups zusätzlich zur Einstiegsfrage
TOPICS_SPEC = [
    {"name": "Book",    "budget": 2, "focus": "books that shape strategy/AI thinking"},
    {"name": "Podcast", "budget": 2, "focus": "audio/video shows that inspire your work"},
    {"name": "Person",  "budget": 2, "focus": "mentors, thinkers, role models"},
    {"name": "Tool",    "budget": 2, "focus": "tools/frameworks used in Data & AI Business Design"},
    {"name": "Idea",    "budget": 2, "focus": "contrarian/interesting concepts you explore"},
]

GLOBAL_QUESTION_CAP = 12  # Sicherheitslimit

# Modelle (per Env/Secrets überschreibbar)
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL      = os.getenv("OPENAI_CHAT_MODEL", "gpt-5")  # Director / Extractor / Vision

# UI-Flags
SHOW_DEBUG = False  # True für Debug-Blöcke

# =======================
# SYSTEM PROMPTS
# =======================
DIRECTOR_SYSTEM_PROMPT = (
    "You are a warm, curious interviewer crafting an 'Expert Card' about the user.\n"
    "Keep it personable; orient toward strategy and Data & AI Business Design when relevant, but don't force it.\n"
    "Rules:\n"
    "- Topic order is a default: Book → Podcast → Person → Tool → Idea. Skip a topic gracefully if the user has nothing.\n"
    "- Ask EXACTLY ONE short question per turn (never multiple). Avoid confirmations.\n"
    "- Vary angles (what/why/example/impact) with 1–2 follow-ups per topic.\n"
    "- Move on once you have enough for the topic.\n"
    "- Stop when topics are finished or the global cap is reached. Then thank the user succinctly.\n"
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
    "Output 1–3 lines. Each line EXACTLY:\n"
    "IMAGE: <direct_image_url> | SOURCE: <source_page_url>\n"
    "No extra commentary."
)

VISION_PICKER_PROMPT = (
    "You are a careful visual validator.\n"
    "Pick the SINGLE best candidate URL from the list that matches the item (book cover / podcast art / person portrait / tool logo).\n"
    "Prefer official or well-known sources, clarity, and correctness.\n"
    "Output EXACTLY the chosen IMAGE url (no comments). If none suitable, output: NONE"
)

# =======================
# CLIENT
# =======================
def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it in Streamlit → Settings → Secrets.")
    return OpenAI(api_key=OPENAI_API_KEY)

# =======================
# UTILS
# =======================
def is_http_url(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not (s.startswith("http://") or s.startswith("https://")):
        return False
    try:
        p = urllib.parse.urlparse(s)
        return bool(p.scheme and p.netloc)
    except Exception:
        return False

def enforce_single_question(text: str) -> str:
    if not text:
        return text
    # Nimm nur die erste Frage/Satz bis zum ersten Fragezeichen/Zeilenumbruch/Punkt (kurz halten)
    m = re.split(r"[\n\r]|[?।。！？!?]|(?<=\.)\s", text.strip(), maxsplit=1)
    return (m[0] + ("?" if "?" in text and not m[0].endswith("?") else "")).strip()

# =======================
# OPENAI CALLS
# =======================
def chat_once(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    r = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return r.choices[0].message.content.strip()

def director_next_question(client: OpenAI, visible_history: List[Dict[str, str]]) -> str:
    msgs = [{"role": "system", "content": DIRECTOR_SYSTEM_PROMPT}] + visible_history[-12:]
    txt = chat_once(client, msgs)
    return enforce_single_question(txt)

def extractor_pull_fields(client: OpenAI, topic_name: str, latest_user_text: str, recent_context: str) -> Dict[str, str]:
    # Liefert ein dict je Topic
    user = f"CURRENT TOPIC: {topic_name}\nContext: {recent_context}\nUser said: {latest_user_text}\nExtract:"
    txt = chat_once(client, [{"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT}, {"role": "user", "content": user}]).strip()
    if txt == "—":
        return {}
    # Parsing je Topic
    if topic_name.lower() == "book":
        # "<title> | <author?>"  (author optional)
        parts = [p.strip() for p in txt.split("|")]
        title = parts[0] if parts else ""
        author = parts[1] if len(parts) > 1 else ""
        return {"title": title, "author": author}
    else:
        name = txt.strip()
        return {"name": name}

def responses_web_search(client: OpenAI, query_block: str) -> str:
    # Hier nutzen wir chat.completions; das Modell soll intern bei Bedarf Webzugriff nutzen.
    msgs = [{"role": "user", "content": query_block}]
    return chat_once(client, msgs)

def vision_pick_best(client: OpenAI, item_desc: str, candidates: List[str]) -> str:
    if not candidates:
        return ""
    # Kompakter Prompt
    prompt = VISION_PICKER_PROMPT + "\n\nITEM:\n" + item_desc + "\n\nCANDIDATES:\n" + "\n".join(candidates)
    txt = chat_once(client, [{"role": "user", "content": prompt}]).strip()
    if txt.upper() == "NONE":
        return ""
    # nimm erste URL im Text
    # Formate sind "IMAGE: <url> | SOURCE: <url>" – wir ziehen die erste http-URL
    m = re.search(r"https?://\S+", txt)
    return m.group(0) if m else ""

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

            # Kandidaten parsen: akzeptiere "IMAGE:" oder "CANDIDATE:" und clamp auf 3
            lines = []
            for ln in (txt or "").splitlines():
                s = ln.strip()
                if not s:
                    continue
                if not (s.upper().startswith("IMAGE:") or s.upper().startswith("CANDIDATE:")):
                    continue
                payload = s.split(":", 1)[1].strip() if ":" in s else s
                parts = [p.strip() for p in payload.split("|")]
                if not parts:
                    continue
                image_url = parts[0]
                page_url = ""
                for p in parts[1:]:
                    if p.lower().startswith("source:"):
                        page_url = p.split(":", 1)[1].strip()
                if is_http_url(image_url):
                    item = f"IMAGE: {image_url} | SOURCE: {page_url}"
                    lines.append(item)
                if len(lines) >= 3:
                    break

            best = vision_pick_best(client, item_desc, lines) if lines else ""
            if best and not is_http_url(best):
                best = ""

            out_q.put({
                "ok": True,
                "job_id": job["job_id"],
                "kind": kind,
                "candidates": lines[:3],
                "best": best or ""
            })
        except Exception as e:
            out_q.put({"ok": False, "job_id": job.get("job_id",""), "error": str(e)})
        finally:
            in_q.task_done()

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
                        t["research"] = t.get("research") or {}
                        t["research"]["candidates"] = res.get("candidates") or []
                        t["research"]["best"] = res.get("best") or ""
                        t["job_inflight"] = False
                        changed = True
                        break
            else:
                st.session_state.debug_log.append(f"[BG ERROR] {res.get('error')}")
        finally:
            st.session_state.bg_result_queue.task_done()
    return changed

# =======================
# STATE / FLOW
# =======================
def init_state():
    if "history" not in st.session_state:
        st.session_state.history: List[Dict[str, str]] = [
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi! I'll help you craft a tiny expert card. First, which book has helped you professionally? Please share the title (author optional)."},
        ]
    if "profile" not in st.session_state:
        st.session_state.profile = {
            "topics": [
                {"name": t["name"], "status": "queued" if i>0 else "active", "budget": t["budget"],
                 "answers": [], "research": {}, "job_inflight": False, "field": {}}
                for i, t in enumerate(TOPICS_SPEC)
            ],
            "current_topic_index": 0,
            "bot_questions": 1,   # wir haben die Startfrage bereits gestellt
            "finalized": False,
        }
    if "bg_task_queue" not in st.session_state:
        st.session_state.bg_task_queue = queue.Queue()
    if "bg_result_queue" not in st.session_state:
        st.session_state.bg_result_queue = queue.Queue()
    if "bg_thread" not in st.session_state:
        st.session_state.bg_thread = None
    if "debug_log" not in st.session_state:
        st.session_state.debug_log = []

def current_topic(profile: Dict[str, Any]) -> Dict[str, Any]:
    return profile["topics"][profile["current_topic_index"]]

def advance_topic(profile: Dict[str, Any]):
    # mark current done
    i = profile["current_topic_index"]
    if profile["topics"][i]["status"] != "done":
        profile["topics"][i]["status"] = "done"
    # find next queued
    for j in range(i+1, len(profile["topics"])):
        if profile["topics"][j]["status"] in ("queued",):
            profile["topics"][j]["status"] = "active"
            profile["current_topic_index"] = j
            return
    # none left

def can_ask_more(profile: Dict[str, Any]) -> bool:
    return profile["bot_questions"] < GLOBAL_QUESTION_CAP

def all_topics_done(profile: Dict[str, Any]) -> bool:
    return all(t["status"] == "done" for t in profile["topics"])

def render_timeline(profile: Dict[str, Any]):
    cols = st.columns(len(profile["topics"]))
    for i, t in enumerate(profile["topics"]):
        with cols[i]:
            cur = " (current)" if i == profile["current_topic_index"] and t["status"] == "active" else ""
            st.markdown(f"**{t['name']}**{cur}")
            st.caption(f"status: {t['status']} • followups left: {t['budget']}")
            best_url = (t.get("research") or {}).get("best") or ""
            if is_http_url(best_url):
                st.image(best_url, caption="Selected", use_container_width=True)
            elif best_url:
                st.caption(f"Image note: {best_url}")

# Enqueue-Suche für GENAU das Topic zum Zeitpunkt der Eingabe
def enqueue_search_if_ready(client: OpenAI, topic_index: int, user_text: str):
    topics = st.session_state.profile["topics"]
    if topic_index < 0 or topic_index >= len(topics):
        return
    topic = topics[topic_index]

    research = topic.get("research") or {}
    if topic.get("job_inflight") or research.get("best") or research.get("candidates"):
        return

    # Nur lokale, letzte Historie als Kontext
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

    if (job.get("title") or job.get("name")):
        st.session_state.bg_task_queue.put(job)
        topic["job_inflight"] = True
        if SHOW_DEBUG:
            st.session_state.debug_log.append(f"[ENQ] t#{topic_index} {job}")

# =======================
# STREAMLIT UI
# =======================
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
st.caption("Director (GPT-5 Chat) → Extraction → Web Image Search (LLM-assisted) → Vision pick → Final table")

init_state()

# OpenAI client
client = None
agent_ready = True
try:
    client = get_client()
except Exception as e:
    agent_ready = False
    st.error(f"OpenAI key missing or invalid: {e}")

# Start BG thread once
if agent_ready and st.session_state.bg_thread is None:
    def runner():
        bg_worker(client, st.session_state.bg_task_queue, st.session_state.bg_result_queue)
    t = threading.Thread(target=runner, daemon=True)
    t.start()
    st.session_state.bg_thread = t

# Drain background results
if drain_bg_results():
    st.toast("Background research updated.", icon="🔎")

# Fortschritt
done_count = sum(1 for t in st.session_state.profile["topics"] if t["status"] == "done")
total_steps = len(TOPICS_SPEC)  # simple: je Topic 1 Step
st.progress(done_count / total_steps, text=f"Progress: {done_count}/{total_steps}")

if SHOW_DEBUG:
    with st.expander("Debug"):
        st.json(st.session_state.profile)
        st.json(st.session_state.debug_log)

# Timeline
render_timeline(st.session_state.profile)

# Bisherige Konversation (ohne System)
for m in st.session_state.history:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat Input Handling
user_text = st.chat_input("Type your answer…")
if user_text:
    # SNAP: Topic-Index zum Zeitpunkt der Eingabe
    topic_at_input = st.session_state.profile["current_topic_index"]
    st.session_state.history.append({"role": "user", "content": user_text})
    # Antwort in das passende Topic schreiben (zum Zeitpunkt der Eingabe)
    st.session_state.profile["topics"][topic_at_input]["answers"].append(user_text)

    # Suche für dieses Topic anstoßen
    if agent_ready:
        enqueue_search_if_ready(client, topic_at_input, user_text)

    # Danach Direktionslogik (kurze Frage)
    topic_now = current_topic(st.session_state.profile)

    # Wenn aktuellem Topic noch Budget bleibt → eine Frage
    if agent_ready and topic_now["budget"] > 0 and can_ask_more(st.session_state.profile):
        reply = director_next_question(client, [m for m in st.session_state.history if m["role"] != "system"])
        st.session_state.profile["bot_questions"] += 1
        topic_now["budget"] -= 1
        st.session_state.history.append({"role": "assistant", "content": reply})

        # Wenn Budget jetzt leer → Topic abschließen und weiterschalten
        if topic_now["budget"] <= 0:
            topic_now["status"] = "done"
            advance_topic(st.session_state.profile)
    else:
        # Kein Budget oder Cap erreicht → Topic schließen und ggf. weiter
        if topic_now["status"] == "active":
            topic_now["status"] = "done"
            advance_topic(st.session_state.profile)

        # Falls noch Fragen erlaubt und Topics übrig → nächste Frage
        if agent_ready and not all_topics_done(st.session_state.profile) and can_ask_more(st.session_state.profile):
            reply = director_next_question(client, [m for m in st.session_state.history if m["role"] != "system"])
            st.session_state.profile["bot_questions"] += 1
            next_topic = current_topic(st.session_state.profile)
            if next_topic["budget"] > 0:
                next_topic["budget"] -= 1
            st.session_state.history.append({"role": "assistant", "content": reply})

    # Abschluss-Hinweis, wenn alles fertig oder Cap erreicht
    if all_topics_done(st.session_state.profile) or not can_ask_more(st.session_state.profile):
        if not st.session_state.profile.get("finalized"):
            st.session_state.history.append({
                "role": "assistant",
                "content": "Thanks — I’ve got a solid picture now. I’ll assemble your expert card in the background."
            })
            st.session_state.profile["finalized"] = True

    st.rerun()

# Reset & Utilities
c1, c2 = st.columns(2)
with c1:
    if st.button("🔄 Restart"):
        for k in ["history", "profile", "bg_thread", "bg_task_queue", "bg_result_queue", "debug_log"]:
            if k in st.session_state:
                del st.session_state[k]
        init_state()
        st.rerun()
with c2:
    if st.button("🧹 Clear debug"):
        st.session_state.debug_log = []
        st.rerun()
