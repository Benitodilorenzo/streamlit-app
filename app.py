import os, re, threading, queue, uuid, urllib.parse
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI

# =======================
# CONFIG
# =======================
APP_TITLE = "📝 Expert Card Creator"

TOPICS_SPEC = [
    {"name": "Book",    "budget": 2, "focus": "books that shape strategy/AI thinking"},
    {"name": "Podcast", "budget": 2, "focus": "audio/video shows that inspire your work"},
    {"name": "Person",  "budget": 2, "focus": "mentors, thinkers, role models"},
    {"name": "Tool",    "budget": 2, "focus": "tools/frameworks in Data & AI Business Design"},
    {"name": "Idea",    "budget": 2, "focus": "contrarian/interesting concepts you explore"},
]

GLOBAL_QUESTION_CAP = 12  # Sicherheitslimit

# Modelle
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL      = os.getenv("OPENAI_CHAT_MODEL", "gpt-5")  # Director / Extractor / Vision / Finalizer

# UI-Flags
SHOW_DEBUG = False

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

SEARCHER_SYSTEM = (
    "You are a researcher using a web search tool.\n"
    "Find high-quality candidate images that best represent the item.\n"
    "Heuristics:\n"
    "- Book: publisher/Amazon/Open Library/Wikipedia; avoid fan art.\n"
    "- Podcast: official channel art (YouTube/Spotify/Apple) or website.\n"
    "- Person: clear, respectful portrait (Wikipedia/official site preferred).\n"
    "- Tool: official logo/hero image.\n"
    "IMPORTANT: Perform at most 3 focused searches. Stop early if you have 2–3 strong candidates.\n"
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

FINALIZER_SYSTEM_PROMPT = (
    "You turn raw interview answers into concise, upbeat lines for an 'Expert Card'.\n"
    "Style: professional, warm, not sycophantic, 1–2 sentences per item, no fluff, no invention."
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
    # Nimm nur die erste kurze Frage
    parts = re.split(r"[\n\r]|[?]|(?<=\.)\s", text.strip(), maxsplit=1)
    out = parts[0].strip()
    if "?" in text and not out.endswith("?"):
        out += "?"
    return out

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
    user = f"CURRENT TOPIC: {topic_name}\nContext: {recent_context}\nUser said: {latest_user_text}\nExtract:"
    txt = chat_once(client, [
        {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]).strip()
    if txt == "—":
        return {}
    if topic_name.lower() == "book":
        parts = [p.strip() for p in txt.split("|")]
        title = parts[0] if parts else ""
        author = parts[1] if len(parts) > 1 else ""
        return {"title": title, "author": author}
    else:
        return {"name": txt.strip()}

def web_search_block(kind: str, title: str = "", author: str = "", name: str = "") -> str:
    if kind == "book":
        return f"{SEARCHER_SYSTEM}\nQuery: Book cover: \"{title}\" {author}".strip()
    if kind == "podcast":
        return f"{SEARCHER_SYSTEM}\nQuery: Podcast cover art: \"{title}\"".strip()
    if kind == "person":
        return f"{SEARCHER_SYSTEM}\nQuery: Portrait photo: \"{name}\"".strip()
    if kind == "tool":
        return f"{SEARCHER_SYSTEM}\nQuery: Official logo or hero image: \"{title}\"".strip()
    return f"{SEARCHER_SYSTEM}\nQuery: Representative image: \"{title or name}\"".strip()

def responses_web_search(client: OpenAI, query_block: str) -> str:
    # Ohne Responses-Tool: wir nutzen das Chat-Modell (internes Browsing ok, Output-Format ist strikt).
    return chat_once(client, [{"role": "user", "content": query_block}])

def vision_pick_best(client: OpenAI, item_desc: str, candidates: List[str]) -> str:
    if not candidates:
        return ""
    prompt = VISION_PICKER_PROMPT + "\n\nITEM:\n" + item_desc + "\n\nCANDIDATES:\n" + "\n".join(candidates)
    txt = chat_once(client, [{"role": "user", "content": prompt}]).strip()
    if txt.upper() == "NONE":
        return ""
    m = re.search(r"https?://\S+", txt)
    return m.group(0) if m else ""

def finalize_item_text(client: OpenAI, topic_name: str, raw_answers: List[str]) -> str:
    if not raw_answers:
        return ""
    content = (
        f"Topic: {topic_name}\n"
        f"Raw answers (most recent first):\n- " + "\n- ".join(list(reversed(raw_answers))[:4]) + "\n"
        "Write a 1–2 sentence, specific line for the expert card."
    )
    return chat_once(client, [
        {"role": "system", "content": FINALIZER_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ])

# =======================
# BACKGROUND WORKER
# =======================
def bg_worker(client: OpenAI, in_q: "queue.Queue[Dict[str, Any]]", out_q: "queue.Queue[Dict[str, Any]]"):
    while True:
        job = in_q.get()
        if job is None:  # sentinel
            in_q.task_done()
            break
        try:
            kind = job["kind"]
            title = job.get("title", "")
            author = job.get("author", "")
            name  = job.get("name", "")

            qblock = web_search_block(kind, title, author, name)
            txt = responses_web_search(client, qblock)

            lines: List[str] = []
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
                # Optionales SOURCE-Feld normalisieren
                page_url = ""
                for p in parts[1:]:
                    if p.lower().startswith("source:"):
                        page_url = p.split(":", 1)[1].strip()
                if is_http_url(image_url):
                    lines.append(f"IMAGE: {image_url} | SOURCE: {page_url}")
                if len(lines) >= 3:
                    break

            item_desc = f'{kind.title()}: "{title or name}" {author}'.strip()
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
            out_q.put({"ok": False, "job_id": job.get("job_id", ""), "error": str(e)})
        finally:
            in_q.task_done()

def drain_bg_results(profile: Dict[str, Any], out_q: "queue.Queue[Dict[str, Any]]") -> bool:
    changed = False
    while True:
        try:
            res = out_q.get_nowait()
        except queue.Empty:
            break
        try:
            if res.get("ok"):
                kind = res["kind"]
                for t in profile["topics"]:
                    if t["name"].lower() == kind:
                        t["research"] = t.get("research") or {}
                        t["research"]["candidates"] = res.get("candidates") or []
                        t["research"]["best"] = res.get("best") or ""
                        t["job_inflight"] = False
                        changed = True
                        break
        finally:
            out_q.task_done()
    return changed

# =======================
# STATE / FLOW
# =======================
def init_state():
    if "history" not in st.session_state:
        st.session_state.history = [
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
            "bot_questions": 1,   # Startfrage gestellt
            "finalized": False,
            "final_lines": {},    # Topic → finaler Text
        }
    if "bg_task_queue" not in st.session_state:
        st.session_state.bg_task_queue = queue.Queue()
    if "bg_result_queue" not in st.session_state:
        st.session_state.bg_result_queue = queue.Queue()
    if "bg_thread" not in st.session_state:
        # Thread bekommt die QUEUES als feste Referenzen (kein Zugriff auf session_state im Worker)
        client = get_client()
        t = threading.Thread(target=bg_worker, args=(client, st.session_state.bg_task_queue, st.session_state.bg_result_queue), daemon=True)
        t.start()
        st.session_state.bg_thread = t

def current_topic(profile: Dict[str, Any]) -> Dict[str, Any]:
    return profile["topics"][profile["current_topic_index"]]

def advance_topic(profile: Dict[str, Any]):
    i = profile["current_topic_index"]
    if profile["topics"][i]["status"] != "done":
        profile["topics"][i]["status"] = "done"
    for j in range(i+1, len(profile["topics"])):
        if profile["topics"][j]["status"] == "queued":
            profile["topics"][j]["status"] = "active"
            profile["current_topic_index"] = j
            return

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
                # irgendein Text (z. B. 'NONE' oder 'Sorry') – nur als Notiz anzeigen
                st.caption(f"Image note: {best_url}")

# Enqueue-Suche für den Topic-Index zum Eingabezeitpunkt
def enqueue_search_if_ready(client: OpenAI, profile: Dict[str, Any], topic_index: int, user_text: str):
    if topic_index < 0 or topic_index >= len(profile["topics"]):
        return
    topic = profile["topics"][topic_index]
    research = topic.get("research") or {}
    if topic.get("job_inflight") or research.get("best") or research.get("candidates"):
        return

    # kurzer lokaler Kontext
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

def maybe_finalize_texts(client: OpenAI, profile: Dict[str, Any]):
    """Erzeuge pro Topic einen finalen 1–2 Sätze-Text (einmalig)."""
    for t in profile["topics"]:
        nm = t["name"]
        if profile["final_lines"].get(nm):
            continue
        if not t["answers"]:
            continue
        try:
            line = finalize_item_text(client, nm, t["answers"])
            profile["final_lines"][nm] = line
        except Exception:
            profile["final_lines"][nm] = ""

def render_final_table(profile: Dict[str, Any]):
    if not profile.get("finalized"):
        return
    # Wenn überhaupt keine finalen Zeilen existieren, nichts rendern
    if not any(profile["final_lines"].get(t["name"], "").strip() for t in profile["topics"]):
        return

    st.subheader("Your Expert Card")
    for t in profile["topics"]:
        name = t["name"]
        text = (profile["final_lines"].get(name, "") or "").strip()
        best_url = (t.get("research") or {}).get("best") or ""
        cols = st.columns([1, 2])
        with cols[0]:
            if is_http_url(best_url):
                st.image(best_url, use_container_width=True)
            else:
                st.caption("(image pending)")
        with cols[1]:
            st.markdown(f"**{name}**")
            if text:
                st.write(text)
            else:
                st.caption("(text pending)")

# =======================
# STREAMLIT UI
# =======================
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
st.caption("Director (GPT-5 Chat) → Extraction → Web Image Search → Vision pick → Final table")

# INIT
init_state()
client = get_client()

# Drain BG results
if drain_bg_results(st.session_state.profile, st.session_state.bg_result_queue):
    st.toast("Background research updated.", icon="🔎")

# Fortschritt (je Topic ein Schritt)
done_count = sum(1 for t in st.session_state.profile["topics"] if t["status"] == "done")
total_steps = len(TOPICS_SPEC)
st.progress(done_count / total_steps, text=f"Progress: {done_count}/{total_steps}")

if SHOW_DEBUG:
    with st.expander("Debug"):
        st.json(st.session_state.profile)

# Timeline
render_timeline(st.session_state.profile)

# Chatverlauf (ohne System)
for m in st.session_state.history:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Eingabe
user_text = st.chat_input("Type your answer…")
if user_text:
    # SNAP Topic zum Zeitpunkt der Eingabe
    topic_at_input = st.session_state.profile["current_topic_index"]
    st.session_state.history.append({"role": "user", "content": user_text})
    st.session_state.profile["topics"][topic_at_input]["answers"].append(user_text)

    # Suche anstoßen
    enqueue_search_if_ready(client, st.session_state.profile, topic_at_input, user_text)

    # Director
    topic_now = st.session_state.profile["topics"][st.session_state.profile["current_topic_index"]]
    if topic_now["budget"] > 0 and can_ask_more(st.session_state.profile):
        reply = director_next_question(client, [m for m in st.session_state.history if m["role"] != "system"])
        st.session_state.profile["bot_questions"] += 1
        topic_now["budget"] -= 1
        st.session_state.history.append({"role": "assistant", "content": reply})
        if topic_now["budget"] <= 0:
            topic_now["status"] = "done"
            advance_topic(st.session_state.profile)
    else:
        if topic_now["status"] == "active":
            topic_now["status"] = "done"
            advance_topic(st.session_state.profile)
        if not all_topics_done(st.session_state.profile) and can_ask_more(st.session_state.profile):
            reply = director_next_question(client, [m for m in st.session_state.history if m["role"] != "system"])
            st.session_state.profile["bot_questions"] += 1
            next_topic = st.session_state.profile["topics"][st.session_state.profile["current_topic_index"]]
            if next_topic["budget"] > 0:
                next_topic["budget"] -= 1
            st.session_state.history.append({"role": "assistant", "content": reply})

    # Finalisierung, wenn fertig oder Cap erreicht
    if all_topics_done(st.session_state.profile) or not can_ask_more(st.session_state.profile):
        if not st.session_state.profile.get("finalized"):
            st.session_state.history.append({"role": "assistant", "content": "Thanks — I’ve got a solid picture now. I’ll assemble your expert card in the background."})
            st.session_state.profile["finalized"] = True
        # Sofort die Textteile erzeugen (Bilder kommen ggf. später)
        maybe_finalize_texts(client, st.session_state.profile)

    st.rerun()

# Wenn finalisiert, zeige Steckbrief (Bilder werden dynamisch nachgeladen)
if st.session_state.profile.get("finalized"):
    maybe_finalize_texts(client, st.session_state.profile)
    render_final_table(st.session_state.profile)

# Reset / Stop sauber
c1, c2 = st.columns(2)
with c1:
    if st.button("🔄 Restart"):
        # Worker sauber stoppen
        try:
            st.session_state.bg_task_queue.put_nowait(None)  # sentinel
        except Exception:
            pass
        try:
            if st.session_state.bg_thread and st.session_state.bg_thread.is_alive():
                st.session_state.bg_thread.join(timeout=1.0)
        except Exception:
            pass
        # State leeren
        for k in ["history", "profile", "bg_thread", "bg_task_queue", "bg_result_queue"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()
with c2:
    if st.button("🧹 Clear final"):
        if "profile" in st.session_state:
            st.session_state.profile["finalized"] = False
            st.session_state.profile["final_lines"] = {}
        st.rerun()
