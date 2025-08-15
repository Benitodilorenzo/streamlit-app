# app_two_agents_async.py
import os
import re
import io
import json
import uuid
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import requests
import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

# =====================================
# CONFIG
# =====================================
APP_TITLE = "🟡 Expert Card Creator — Two Agents (Async + Real Web Search)"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5")
FINALIZER_MODEL = os.getenv("OPENAI_FINALIZER_MODEL", CHAT_MODEL)

GLOBAL_QUESTION_CAP = 12
MAX_SLOTS = 4
HTTP_TIMEOUT = 12
MEDIA_MAX_WORKERS = 3  # parallel media jobs

SHOW_DEBUG = False

MEDIA_DIR = os.path.join(os.getcwd(), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

USER_AGENT = (
    "ExpertCardBot/1.0 (+https://example.local) "
    "Requests-Streamlit"
)

# =====================================
# PROMPTS (Chat-Agent)
# =====================================
# IMPORTANT: No triple backticks inside this string. Use <JSON> ... </JSON> markers.
CHAT_AGENT_SYSTEM_PROMPT = """
You are the single Interview & Orchestration Agent for creating a 4-point “Expert Card”.

PRIMARY OBJECTIVE
- Deliver exactly 4 labeled items (labels are flexible).
- Each item: a crisp 1–2 sentence final line, grounded in the user’s answers.
- Keep a subtle data & AI strategy/design lens when relevant; never force it.

STYLE
- Warm, professional, specific. No filler.
- Ask EXACTLY ONE short question per turn. No double-questions.

ASYNC MEDIA
- If an item is public (book/podcast/person/tool), request media asynchronously and CONTINUE the interview (do not wait).
- You do not reveal any media operations to the user.

STOP
- Stop after 4 slots or if conversation is exhausted.

HIDDEN MACHINE-ONLY DIRECTIVES (append AFTER your visible sentence/question; never mention them):
<JSON>
{"slot_update": {
  "slot_id": "S1",
  "label": "Must-Read",
  "bullets": ["point 1", "point 2", "point 3"],
  "done": false
}}
</JSON>

<JSON>
{"media_request": {
  "slot_id": "S1",
  "intent": "person|book|podcast|tool|generic",
  "query": "focused query for the canonical image",
  "prefer_domains": ["wikipedia.org","openlibrary.org","spotify.com","itunes.apple.com","amazon.de","m.media-amazon.com","books.google.com","vaahlen.de"]
}}
</JSON>

<JSON>
{"finalize_card": {
  "notes": [{"label":"...","bullets":["...","..."]}, {"label":"...","bullets":["..."]}]
}}
</JSON>

RULES
- Exactly one visible question per turn.
- Never ask for confirmations inside the hidden blocks.
- If the user asks you a question, answer briefly (1–2 sentences) and then continue with one question.
"""

FINALIZER_SYSTEM_PROMPT = """
You turn structured notes into a concise, upbeat Expert Card.
Return 4 labeled items with 1–2 sentences each, specific and grounded in the notes.
No fluff, no invention, professional and warm.
Format:

Label: line

Label: line

Label: line

Label: line
"""

# =====================================
# OPENAI client
# =====================================
def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it in Streamlit → Settings → Secrets.")
    return OpenAI(api_key=OPENAI_API_KEY)

client: Optional[OpenAI] = None

# =====================================
# Helpers
# =====================================
# Extract <JSON> ... </JSON> blocks (multiple) and remove them from the visible text
JSON_FENCE_RE = re.compile(r"<JSON>\s*(\{[\s\S]*?\})\s*</JSON>", re.IGNORECASE)

def extract_hidden_json_blocks(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    directives: List[Dict[str, Any]] = []
    if not text:
        return "", directives
    for m in JSON_FENCE_RE.finditer(text):
        block = m.group(1)
        try:
            directives.append(json.loads(block))
        except Exception:
            # ignore malformed block
            pass
    visible = JSON_FENCE_RE.sub("", text).strip()
    return visible, directives

def enforce_single_question(text: str) -> str:
    """
    Keep only the first sentence/question. If there is a question mark anywhere,
    ensure the visible output ends with a single question mark.
    """
    if not text:
        return ""
    # split by newline or question-mark or sentence end
    parts = re.split(r"[\n\r]|[?]|(?<=\.)\s", text.strip(), maxsplit=1)
    out = parts[0].strip()
    if "?" in text and not out.endswith("?"):
        out += "?"
    return out

def is_http_url(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s2 = s.strip()
    if not (s2.startswith("http://") or s2.startswith("https://")):
        return False
    try:
        p = urllib.parse.urlparse(s2)
        return bool(p.scheme and p.netloc)
    except Exception:
        return False

def http_get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> Optional[dict]:
    try:
        h = {"User-Agent": USER_AGENT}
        if headers:
            h.update(headers)
        r = requests.get(url, params=params, headers=h, timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def score_url(url: str, prefer_domains: Optional[List[str]]) -> int:
    score = 0
    if not url:
        return -999
    if prefer_domains:
        for d in prefer_domains:
            if d and d.lower() in url.lower():
                score += 30
    # rough heuristics
    if "m.media-amazon.com" in url: score += 20
    if "books.google.com" in url: score += 12
    if "openlibrary.org" in url or "covers.openlibrary.org" in url: score += 10
    if url.endswith(".jpg") or url.endswith(".png") or ".jpg" in url or ".png" in url: score += 5
    return score

def save_uploaded_image(file, slot_id: str) -> str:
    try:
        img = Image.open(file)
        path = os.path.join(MEDIA_DIR, f"{slot_id}_upload.png")
        img.save(path, format="PNG")
        return path
    except Exception:
        return ""

def generate_placeholder_icon(text: str, slot_id: str) -> str:
    # simple local icon (no external API)
    size = (640, 640)
    img = Image.new("RGB", size, color=(24, 31, 55))
    draw = ImageDraw.Draw(img)
    # rings
    for r, col in [(260, (57, 96, 199)), (200, (73, 199, 142)), (140, (255, 205, 86))]:
        draw.ellipse([(size[0]//2-r, size[1]//2-r), (size[0]//2+r, size[1]//2+r)], outline=col, width=8)
    # text
    txt = (text or "Idea").strip()[:28]
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
    except Exception:
        font = ImageFont.load_default()
    w, h = draw.textsize(txt, font=font)
    draw.text(((size[0]-w)//2, (size[1]-h)//2), txt, fill=(240, 240, 240), font=font)
    path = os.path.join(MEDIA_DIR, f"{slot_id}_placeholder.png")
    img.save(path, format="PNG")
    return path

# =====================================
# Real web lookups
# =====================================
def search_google_books(query: str) -> List[Dict[str, str]]:
    data = http_get_json(
        "https://www.googleapis.com/books/v1/volumes",
        params={"q": query, "maxResults": 5}
    )
    out: List[Dict[str, str]] = []
    if not data or not data.get("items"):
        return out
    for it in data["items"]:
        info = it.get("volumeInfo", {})
        links = info.get("imageLinks", {}) or {}
        # prefer higher res if available
        for key in ["extraLarge","large","medium","small","thumbnail","smallThumbnail"]:
            url = links.get(key)
            if url:
                if url.startswith("http://"):
                    url = "https://" + url[len("http://"):]
                out.append({
                    "image_url": url,
                    "page_url": info.get("infoLink") or "",
                    "title": info.get("title","")
                })
                break
        if len(out) >= 3:
            break
    return out

def search_openlibrary(query: str) -> List[Dict[str, str]]:
    data = http_get_json("https://openlibrary.org/search.json", params={"q": query, "limit": 5})
    out: List[Dict[str, str]] = []
    if not data or not data.get("docs"):
        return out
    for d in data["docs"]:
        cover_id = d.get("cover_i")
        if not cover_id:
            continue
        img = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
        page = f"https://openlibrary.org{d.get('key','')}"
        out.append({"image_url": img, "page_url": page, "title": d.get("title","")})
        if len(out) >= 3:
            break
    return out

def search_wikipedia_image(query: str, langs: Optional[List[str]] = None) -> List[Dict[str, str]]:
    if not langs:
        langs = ["en", "de"]
    out: List[Dict[str, str]] = []
    for lang in langs:
        base = f"https://{lang}.wikipedia.org"
        s = http_get_json(f"{base}/w/rest.php/v1/search/title", params={"q": query, "limit": 3})
        if not s or not s.get("pages"):
            continue
        for p in s["pages"]:
            title = p.get("title")
            if not title:
                continue
            summ = http_get_json(f"{base}/api/rest_v1/page/summary/{urllib.parse.quote(title)}")
            if not summ:
                continue
            img = (summ.get("originalimage") or {}).get("source") or (summ.get("thumbnail") or {}).get("source")
            page_url = f"{base}/wiki/{urllib.parse.quote(title)}"
            if img and is_http_url(img):
                out.append({"image_url": img, "page_url": page_url, "title": title})
            if len(out) >= 3:
                break
        if out:
            break
    return out

def search_itunes_podcast(query: str) -> List[Dict[str, str]]:
    data = http_get_json(
        "https://itunes.apple.com/search",
        params={"term": query, "media": "podcast", "limit": 5}
    )
    out: List[Dict[str, str]] = []
    if not data or not data.get("results"):
        return out
    for r in data["results"]:
        img = r.get("artworkUrl600") or r.get("artworkUrl100") or r.get("artworkUrl60")
        page = r.get("collectionViewUrl") or r.get("trackViewUrl") or ""
        if img and is_http_url(img):
            out.append({"image_url": img, "page_url": page, "title": r.get("collectionName","")})
        if len(out) >= 3:
            break
    return out

# =====================================
# Master resolver
# =====================================
def resolve_media(intent: str, query: str, prefer_domains: Optional[List[str]] = None) -> Dict[str, Any]:
    candidates: List[Dict[str, str]] = []
    try:
        if intent == "book":
            # Try Google Books first (often exact cover), then OpenLibrary
            candidates = search_google_books(query)
            if len(candidates) < 2:
                ol = search_openlibrary(query)
                for c in ol:
                    if c not in candidates:
                        candidates.append(c)
                        if len(candidates) >= 3:
                            break
        elif intent == "podcast":
            candidates = search_itunes_podcast(query)
        elif intent == "person":
            candidates = search_wikipedia_image(query)
        elif intent == "tool":
            candidates = search_wikipedia_image(query)
        else:
            candidates = []
    except Exception:
        candidates = []

    if candidates:
        ranked = sorted(candidates, key=lambda c: score_url(c.get("image_url",""), prefer_domains), reverse=True)
        best = ranked[0]["image_url"] if ranked else ""
        return {
            "status": "found",
            "best_image_url": best,
            "candidates": ranked[:3],
            "notes": "Resolved via real web sources"
        }

    return {
        "status": "none",
        "best_image_url": "",
        "candidates": [],
        "notes": "No candidates from web sources"
    }

# =====================================
# Async media jobs
# =====================================
def media_job(slot_id: str, intent: str, query: str, prefer_domains: Optional[List[str]] = None) -> Dict[str, Any]:
    res = resolve_media(intent, query, prefer_domains)
    res.update({"slot_id": slot_id, "type": "MEDIA_RESULT"})
    if res["status"] == "none":
        # create a local placeholder to avoid empty UI
        placeholder_path = generate_placeholder_icon(query or intent, slot_id)
        res["status"] = "generated"
        res["best_image_url"] = placeholder_path
        res["notes"] = "Generated local placeholder icon"
    return res

def ensure_executor():
    if "media_executor" not in st.session_state:
        st.session_state.media_executor = ThreadPoolExecutor(max_workers=MEDIA_MAX_WORKERS)
    if "media_futures" not in st.session_state:
        st.session_state.media_futures: Dict[str, Future] = {}

def schedule_media_job(slot_id: str, intent: str, query: str, prefer_domains: Optional[List[str]] = None):
    ensure_executor()
    job_id = str(uuid.uuid4())[:8]
    fut = st.session_state.media_executor.submit(media_job, slot_id, intent, query, prefer_domains)
    st.session_state.media_futures[job_id] = fut
    return job_id

def poll_media_jobs():
    if "media_futures" not in st.session_state:
        return False
    completed_any = False
    to_delete = []
    for job_id, fut in list(st.session_state.media_futures.items()):
        if fut.done():
            completed_any = True
            try:
                res = fut.result()
                apply_media_result(res)
                st.toast(f"Media updated for {res.get('slot_id')}", icon="🖼️")
            except Exception as e:
                st.toast(f"Media job failed: {e}", icon="⚠️")
            to_delete.append(job_id)
    for job_id in to_delete:
        del st.session_state.media_futures[job_id]
    return completed_any

# =====================================
# App state / slots
# =====================================
def init_state():
    if "history" not in st.session_state:
        st.session_state.history = [
            {"role": "system", "content": CHAT_AGENT_SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi! Which single book most changed how you think about building data-driven products? Title only."}
        ]
    if "slots" not in st.session_state:
        st.session_state.slots: List[Dict[str, Any]] = []
    if "finalized" not in st.session_state:
        st.session_state.finalized = False
    if "final_lines" not in st.session_state:
        st.session_state.final_lines: Dict[str, str] = {}
    if "agent_questions" not in st.session_state:
        st.session_state.agent_questions = 1  # greeting counted

def get_slot(slot_id: str) -> Optional[Dict[str, Any]]:
    for s in st.session_state.slots:
        if s.get("slot_id") == slot_id:
            return s
    return None

def upsert_slot(slot_id: str, label: Optional[str] = None, bullets: Optional[List[str]] = None, done: Optional[bool] = None):
    s = get_slot(slot_id)
    if s is None:
        s = {
            "slot_id": slot_id,
            "label": label or "",
            "bullets": bullets or [],
            "done": bool(done),
            "media": {"status": "pending", "best_image_url": "", "candidates": []}
        }
        st.session_state.slots.append(s)
    else:
        if label is not None:
            s["label"] = label
        if bullets is not None:
            s["bullets"] = bullets
        if done is not None:
            s["done"] = bool(done)

def apply_media_result(media_result: Dict[str, Any]):
    slot_id = media_result.get("slot_id")
    s = get_slot(slot_id)
    if not s:
        return
    s["media"]["status"] = media_result.get("status", "none")
    s["media"]["best_image_url"] = media_result.get("best_image_url", "")
    s["media"]["candidates"] = media_result.get("candidates", [])

# =====================================
# Finalization
# =====================================
def finalize_card_from_notes(notes: List[Dict[str, Any]]) -> Dict[str, str]:
    r = client.chat.completions.create(
        model=FINALIZER_MODEL,
        messages=[
            {"role": "system", "content": FINALIZER_SYSTEM_PROMPT},
            {"role": "user", "content": "Create the final Expert Card with 4 labeled items based on these notes:\n" +
             "\n".join(f"- {n['label']}: " + "; ".join(n.get('bullets', [])) for n in notes)}
        ]
    )
    txt = r.choices[0].message.content or ""
    lines = [l.strip("- ").strip() for l in txt.splitlines() if l.strip()]
    out: Dict[str, str] = {}
    for l in lines:
        if ":" in l:
            label, body = l.split(":", 1)
            out[label.strip()] = body.strip()
    return out

# =====================================
# Chat loop (no blocking tools; hidden JSON directives)
# =====================================
def chat_once(messages: List[Dict[str, Any]]) -> str:
    r = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return r.choices[0].message.content or ""

def run_chat_turn(user_text: Optional[str]):
    if user_text:
        st.session_state.history.append({"role": "user", "content": user_text})

    # model response
    msg = chat_once(st.session_state.history)
    visible, directives = extract_hidden_json_blocks(msg)
    visible = enforce_single_question(visible)

    # append visible assistant content
    st.session_state.history.append({"role": "assistant", "content": visible})
    if visible.endswith("?"):
        st.session_state.agent_questions += 1

    # handle directives (async / no blocking)
    for d in directives:
        if "slot_update" in d:
            su = d["slot_update"]
            upsert_slot(
                slot_id=su.get("slot_id", str(uuid.uuid4())[:6]),
                label=su.get("label"),
                bullets=[b for b in su.get("bullets", []) if isinstance(b, str)],
                done=su.get("done", False),
            )
        if "media_request" in d:
            mr = d["media_request"]
            slot_id = mr.get("slot_id", str(uuid.uuid4())[:6])
            upsert_slot(slot_id)  # ensure it exists
            s = get_slot(slot_id)
            if s:
                s["media"]["status"] = "searching"
            schedule_media_job(
                slot_id=slot_id,
                intent=mr.get("intent", "generic"),
                query=mr.get("query", ""),
                prefer_domains=mr.get("prefer_domains", []),
            )
        if "finalize_card" in d:
            notes = d["finalize_card"].get("notes", [])
            if notes:
                st.session_state.final_lines = finalize_card_from_notes(notes)
                st.session_state.finalized = True

# =====================================
# UI
# =====================================
def render_progress():
    filled = sum(1 for s in st.session_state.slots if s.get("label") and s.get("bullets"))
    st.progress(min(1.0, filled / MAX_SLOTS), text=f"Progress: {filled}/{MAX_SLOTS}")

def render_timeline():
    slots = st.session_state.slots
    if not slots:
        return
    cols = st.columns(min(len(slots), 4))
    for i, s in enumerate(slots[:4]):
        with cols[i]:
            label = s.get("label", "(pending)") or "(pending)"
            st.markdown(f"**{label}**")
            st.caption("status: " + (s.get("media", {}).get("status") or "pending"))
            best = (s.get("media", {}).get("best_image_url") or "").strip()
            if best:
                try:
                    st.image(best, use_container_width=True)
                except Exception:
                    st.caption("(image unavailable)")
            else:
                st.caption("(image pending)")

def render_chat():
    for m in st.session_state.history:
        if m["role"] == "system":
            continue
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def render_final_card():
    if not st.session_state.finalized or not st.session_state.final_lines:
        return
    st.subheader("Your Expert Card")
    for s in st.session_state.slots[:MAX_SLOTS]:
        label = s.get("label", "").strip()
        if not label:
            continue
        line = st.session_state.final_lines.get(label, "").strip()
        if not line:
            continue
        best = (s.get("media", {}).get("best_image_url") or "").strip()
        cols = st.columns([1, 2])
        with cols[0]:
            if best:
                try:
                    st.image(best, use_container_width=True)
                except Exception:
                    st.caption("(image unavailable)")
            else:
                st.caption("(image pending)")
        with cols[1]:
            st.markdown(f"**{label}**")
            st.write(line)

def render_overrides():
    st.markdown("### Media overrides (optional)")
    for s in st.session_state.slots:
        with st.expander(f"Override image for {s.get('label') or s['slot_id']}"):
            col1, col2 = st.columns(2)
            with col1:
                url = st.text_input("Image URL (http/https)", key=f"url_{s['slot_id']}")
                if url and is_http_url(url):
                    s["media"]["status"] = "found"
                    s["media"]["best_image_url"] = url
                    st.success("Using URL override.")
            with col2:
                up = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key=f"up_{s['slot_id']}")
                if up:
                    path = save_uploaded_image(up, s["slot_id"])
                    if path:
                        s["media"]["status"] = "uploaded"
                        s["media"]["best_image_url"] = path
                        st.success("Using uploaded image.")

# =====================================
# MAIN
# =====================================
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
st.caption("Chat continues while media searches run in parallel (OpenLibrary, Google Books, Wikipedia, iTunes).")

client = get_client()
init_state()
ensure_executor()

# Poll background jobs at the start of each run
if poll_media_jobs():
    pass  # state updated

render_progress()
render_timeline()
render_chat()

# Chat input
user_text = st.chat_input("Type your answer…")
if user_text:
    run_chat_turn(user_text)
    st.rerun()

# Final card (if available)
render_final_card()

# Media overrides UI
render_overrides()

# Controls
colA, colB = st.columns(2)
with colA:
    if st.button("🔄 Restart"):
        # shutdown executor
        try:
            if "media_executor" in st.session_state:
                st.session_state.media_executor.shutdown(cancel_futures=True)
        except Exception:
            pass
        for k in ["history","slots","finalized","final_lines","agent_questions","media_executor","media_futures"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()
with colB:
    if st.button("🧹 Clear final"):
        st.session_state.finalized = False
        st.session_state.final_lines = {}
        st.rerun()

# Debug
if SHOW_DEBUG:
    with st.expander("Debug"):
        st.json({
            "slots": st.session_state.slots,
            "finalized": st.session_state.finalized,
            "final_lines": st.session_state.final_lines,
        })
