# app_expert_card_gpt5.py
# Single-file, production-ready Streamlit app
# Architecture: GPT-5 Chat Agent (interview) + Hosted Web Search (OpenAI Responses API) + Finalizer
# - GPT-5 only (no other models), minimal prompts, no temperature/think settings
# - Hosted Web Search (no own Google/Bing keys). Prefer official domains via instructions.
# - Fire-and-forget media jobs (ThreadPoolExecutor). Chat stays responsive.
# - Entity extraction & insight distillation via compact structured calls.
# - Always show one short visible question per turn (even if tools run).
# - Robust session init, idempotent scheduling, placeholder images, URL/upload overrides.
#
# Requirements:
#   pip install streamlit openai pillow requests
# Environment:
#   OPENAI_API_KEY (required)
# Optional:
#   set STREAMLIT_SERVER_PORT, etc.

import os, re, uuid, urllib.parse, json, time
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

# =========================
# CONFIG
# =========================
APP_TITLE = "🟡 Expert Card — GPT-5 (Hosted Web Search, Async, Single File)"
MODEL_SNAPSHOT = os.getenv("OPENAI_GPT5_SNAPSHOT", "gpt-5-2025-08-07")  # pin for stable behavior
OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", MODEL_SNAPSHOT)            # GPT-5 only

HTTP_TIMEOUT = 10
MEDIA_MAX_WORKERS = 4
PLACEHOLDER_MIN_TEXT = 3

MEDIA_DIR = os.path.join(os.getcwd(), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)
UA = "ExpertCard/1.0 (+https://local.app) Streamlit"

# Prefer official/authoritative sources for images
PREFERRED_DOMAINS = {
    "book":    ["m.media-amazon.com", "amazon.de", "books.google.com", "openlibrary.org", "wikipedia.org"],
    "podcast": ["podcasts.apple.com", "itunes.apple.com", "spotify.com"],
    "person":  ["wikipedia.org", "wikidata.org"],
    "tool":    ["wikipedia.org", "docs.", "developer."],
    "film":    ["wikipedia.org", "themoviedb.org", "imdb.com"]
}

# =========================
# OPENAI CLIENT
# =========================
def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    return OpenAI(api_key=key)

client: Optional[OpenAI] = None

# =========================
# SESSION / STATE
# =========================
def init_state():
    if st.session_state.get("initialized"):
        return
    st.session_state.initialized = True

    # dialog history for UI (assistant/user only)
    st.session_state.history: List[Dict[str, Any]] = []

    # slots: up to 4 (S1..S4)
    st.session_state.slots: Dict[str, Dict[str, Any]] = {}
    st.session_state.slot_order = ["S1", "S2", "S3", "S4"]
    st.session_state.current_slot_ix = 0

    # dedupe for media jobs (entity_type+entity_name)
    st.session_state.media_fingerprints: set = set()

    # async executor & jobs
    st.session_state.executor = ThreadPoolExecutor(max_workers=MEDIA_MAX_WORKERS)
    st.session_state.media_jobs: Dict[str, Tuple[str, Future]] = {}

    # finalized lines
    st.session_state.final_lines: Dict[str, str] = {}

def current_slot_id() -> str:
    ix = st.session_state.current_slot_ix
    return st.session_state.slot_order[min(ix, 3)]

def advance_slot():
    st.session_state.current_slot_ix = min(st.session_state.current_slot_ix + 1, 3)

# =========================
# UTILS
# =========================
def _get(url, params=None, headers=None, timeout=HTTP_TIMEOUT):
    h = {"User-Agent": UA}
    if headers: h.update(headers)
    return requests.get(url, params=params, headers=h, timeout=timeout)

def generate_placeholder_icon(text: str, slot_id: str) -> str:
    size = (640, 640)
    img = Image.new("RGB", size, color=(24, 31, 55))
    draw = ImageDraw.Draw(img)
    for r, col in [(260, (57, 96, 199)), (200, (73, 199, 142)), (140, (255, 205, 86))]:
        draw.ellipse([(size[0]//2 - r, size[1]//2 - r), (size[0]//2 + r, size[1]//2 + r)], outline=col, width=8)
    txt = (text or "Idea").strip()
    if len(txt) < PLACEHOLDER_MIN_TEXT: txt = "Idea"
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
    except Exception:
        font = ImageFont.load_default()
    w, h = draw.textsize(txt, font=font)
    draw.text(((size[0] - w)//2, (size[1] - h)//2), txt, fill=(240, 240, 240), font=font)
    path = os.path.join(MEDIA_DIR, f"{slot_id}_ph.png")
    img.save(path, format="PNG")
    return path

def save_uploaded_image(file, slot_id: str) -> str:
    try:
        img = Image.open(file)
        path = os.path.join(MEDIA_DIR, f"{slot_id}_upload.png")
        img.save(path, format="PNG")
        return path
    except Exception:
        return ""

def normalize_entity(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to detect a public entity & normalize to (type, name).
    Returns (entity_type, entity_name) or (None, None).
    Heuristics + simple patterns; backed by GPT router (below).
    """
    s = (raw or "").strip()
    if not s: return (None, None)

    # Quick patterns for book title + author
    m = re.search(r'["“]?([^"\n]+?)["”]?\s+(?:by|von|from)\s+([A-Za-zÀ-ÖØ-öø-ÿ\.\-\s]+)$', s, re.I)
    if m:
        title = m.group(1).strip()
        author = m.group(2).strip()
        return ("book", f"{title} — {author}")

    # If looks like a film
    if re.search(r'\bmovie|film|poster\b', s, re.I):
        return ("film", s.replace("poster", "").strip(' "'))

    # If it looks like a person "Firstname Lastname"
    if len(s.split()) in (2, 3) and s[0].isupper():
        # ambiguous—leave to router
        return (None, None)

    return (None, None)

# =========================
# GPT-5 HELPERS (Responses API)
# =========================

# Hosted Web Search result schema for a single best image
BEST_IMAGE_SCHEMA = {
    "name": "BestImage",
    "schema": {
        "type": "object",
        "properties": {
            "url":        {"type":"string", "description":"Direct image URL (.jpg/.png), not HTML page"},
            "page_url":   {"type":"string", "description":"Canonical source page"},
            "source":     {"type":"string", "description":"Hostname of source"},
            "confidence": {"type":"number"},
            "reason":     {"type":"string"}
        },
        "required": ["url", "page_url"]
    },
    "strict": True
}

# Entity Router schema
ENTITY_SCHEMA = {
    "name": "EntityPick",
    "schema": {
        "type": "object",
        "properties": {
            "detected": {"type":"boolean"},
            "entity_type": {"type":"string", "enum":["book","podcast","person","tool","film","none"]},
            "entity_name": {"type":"string", "description":"Human-readable; for books prefer 'Title — Author' if possible"}
        },
        "required": ["detected","entity_type","entity_name"]
    },
    "strict": True
}

# Insight distillation schema
INSIGHT_SCHEMA = {
    "name": "InsightBullets",
    "schema": {
        "type":"object",
        "properties":{
            "label":{"type":"string"},
            "bullets":{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":4}
        },
        "required":["label","bullets"]
    },
    "strict": True
}

def hosted_search_best_image(entity_type: str, entity_name: str) -> Dict[str, Any]:
    """
    Use GPT-5 Responses API with hosted web_search to pick ONE authoritative image.
    """
    prefer = PREFERRED_DOMAINS.get(entity_type, [])
    instructions = (
        "Find ONE authoritative image for the given public item.\n"
        f"Item type: {entity_type}\n"
        "Requirements:\n"
        f"- Prefer domains: {', '.join(prefer) if prefer else 'no preference'}\n"
        "- MUST return a direct image URL (.jpg or .png), not a page.\n"
        "- Avoid thumbnails, memes, watermarks, wrong items.\n"
        "- If multiple candidates: pick official/primary source, large resolution, correct match.\n"
        "Return structured JSON only."
    )
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            tools=[{"type":"web_search"}],
            tool_choice="auto",
            reasoning={"effort":"minimal"},
            verbosity="low",
            response_format={"type":"json_schema","json_schema": BEST_IMAGE_SCHEMA},
            input=[{
                "role":"user",
                "content":[
                    {"type":"text","text":instructions},
                    {"type":"input_text","text": f"Item: {entity_name}"}
                ]
            }]
        )
        data = getattr(resp, "output_parsed", None) or getattr(resp, "output", None)
        if isinstance(data, dict) and data.get("url") and data.get("page_url"):
            return {"status":"found","best_image_url": data["url"], "candidates":[data], "notes": data.get("reason","")}
    except Exception:
        pass
    return {"status":"none","best_image_url":"","candidates":[],"notes":"no result"}

def route_entity_with_gpt(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Lightweight router: ask GPT-5 to classify and produce normalized name.
    """
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            reasoning={"effort":"minimal"},
            verbosity="low",
            response_format={"type":"json_schema","json_schema": ENTITY_SCHEMA},
            input=[{"role":"user","content":[{"type":"text","text": f"Extract one public entity if present from: {text}"}]}]
        )
        data = getattr(resp, "output_parsed", None) or getattr(resp, "output", None)
        if isinstance(data, dict) and data.get("detected") and data.get("entity_type") != "none":
            return (data.get("entity_type"), data.get("entity_name"))
    except Exception:
        pass
    return (None, None)

def distill_insights_with_gpt(text: str, context_hint: str = "") -> Tuple[Optional[str], List[str]]:
    """
    Extract a short label + 2–4 concise bullets from user's explanation.
    """
    try:
        inst = (
            "From the user's message, extract one concise label and 2–4 crisp bullets (facts/insights). "
            "No fluff, keep it specific. If context mentions a book/person/tool, you may reference it once."
        )
        if context_hint:
            inst += f" Context: {context_hint}"
        resp = client.responses.create(
            model=OPENAI_MODEL,
            reasoning={"effort":"minimal"},
            verbosity="low",
            response_format={"type":"json_schema","json_schema": INSIGHT_SCHEMA},
            input=[{"role":"user","content":[{"type":"text","text": inst}, {"type":"input_text","text": text}]}]
        )
        data = getattr(resp, "output_parsed", None) or getattr(resp, "output", None)
        if isinstance(data, dict) and data.get("label") and data.get("bullets"):
            bullets = [b for b in data["bullets"] if isinstance(b,str)]
            return (data["label"], bullets[:4])
    except Exception:
        pass
    return (None, [])

def finalize_card_with_gpt(notes: List[Dict[str, Any]]) -> Dict[str, str]:
    # Strict, minimal finalizer (4 labeled lines)
    prompt = (
        "Create a concise Expert Card with exactly 4 labeled items.\n"
        "Each item: 1–2 short sentences, grounded in bullets; no fluff.\n"
        "Return plain text, lines start with '- Label: text'"
    )
    try:
        txt = "\n".join(f"- {n['label']}: " + "; ".join(n.get("bullets", [])) for n in notes)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content": prompt},
                {"role":"user","content": txt}
            ]
        )
        out = (resp.choices[0].message.content or "").strip()
        lines = [l.strip("- ").strip() for l in out.splitlines() if l.strip()]
        res: Dict[str,str] = {}
        for l in lines:
            if ":" in l:
                lab, body = l.split(":",1)
                res[lab.strip()] = body.strip()
        return res
    except Exception:
        return {}

# =========================
# CHAT AGENT (One question per turn)
# =========================
CHAT_POLICY = (
    "ROLE\n"
    "You are a concise interviewer. Goals:\n"
    "1) Identify PUBLIC items (book, podcast, person, tool, film).\n"
    "2) Get the user's personal insight/practice about them.\n\n"
    "RULES\n"
    "- Exactly ONE short question per turn.\n"
    "- If the user mentioned a PUBLIC item, do not ask for confirmation.\n"
    "- Keep it warm, specific, professional; no fluff.\n"
)

def state_summary() -> str:
    parts = []
    for sid in st.session_state.slot_order:
        s = st.session_state.slots.get(sid)
        if not s:
            parts.append(f"{sid}: (empty)")
            continue
        done = "done" if s.get("done") else "open"
        parts.append(f"{sid}: {s.get('label','(pending)')} [{done}]")
    return "STATE\n" + " | ".join(parts) + f"\nCURRENT_SLOT_ID={current_slot_id()}"

def chat_reply(last_user_text: Optional[str]) -> str:
    msgs = [
        {"role":"system","content": CHAT_POLICY},
        {"role":"system","content": state_summary()}
    ]
    # last 6 turns only
    tail = [m for m in st.session_state.history if m["role"] in ("user","assistant")]
    tail = tail[-6:]
    msgs.extend(tail)
    if last_user_text:
        msgs.append({"role":"user","content": last_user_text})

    # One-shot completion; keep it snappy
    res = client.chat.completions.create(model=OPENAI_MODEL, messages=msgs)
    text = (res.choices[0].message.content or "").strip()
    # Guarantee one visible question
    if not text or not text.strip().endswith("?"):
        # Fallback by type hint (best effort)
        text = "What changed in your practice after this?"
    return text

# =========================
# ORCHESTRATOR
# =========================
class Orchestrator:
    def __init__(self):
        self.slots = st.session_state.slots
        self.media_jobs = st.session_state.media_jobs
        self.executor: ThreadPoolExecutor = st.session_state.executor
        self.fp = st.session_state.media_fingerprints

    def upsert_slot(self, slot_id: str, label: Optional[str] = None,
                    bullets: Optional[List[str]] = None, done: Optional[bool] = None):
        s = self.slots.get(slot_id, {
            "slot_id": slot_id,
            "label": "",
            "bullets": [],
            "done": False,
            "media": {"status":"pending","best_image_url":"","candidates":[]}
        })
        if label is not None: s["label"] = label[:80]
        if bullets is not None: s["bullets"] = [b.strip() for b in bullets][:4]
        if done is not None: s["done"] = bool(done)
        self.slots[slot_id] = s

    def schedule_media_search(self, slot_id: str, entity_type: str, entity_name: str):
        key = f"{entity_type}|{entity_name}".lower().strip()
        if key in self.fp:
            return  # already scheduled
        self.fp.add(key)
        s = self.slots.get(slot_id) or {"slot_id": slot_id, "label":"", "bullets":[], "done": False, "media":{}}
        s["media"] = {"status":"searching","best_image_url":"","candidates":[]}
        self.slots[slot_id] = s
        fut = self.executor.submit(self._media_job, slot_id, entity_type, entity_name)
        job_id = str(uuid.uuid4())[:8]
        self.media_jobs[job_id] = (slot_id, fut)
        st.toast("🔎 Image search started…", icon="🖼️")

    def _media_job(self, slot_id: str, entity_type: str, entity_name: str) -> Dict[str, Any]:
        res = hosted_search_best_image(entity_type, entity_name)
        if res.get("status") != "found":
            # fallback placeholder
            res = {
                "status":"generated",
                "best_image_url": generate_placeholder_icon(entity_name or entity_type, slot_id),
                "candidates": [],
                "notes": "placeholder"
            }
        res["slot_id"] = slot_id
        return res

    def poll_media(self) -> List[str]:
        updated = []
        to_delete = []
        for job_id, (slot_id, fut) in list(self.media_jobs.items()):
            if fut.done():
                try:
                    res = fut.result()
                    s = self.slots.get(slot_id)
                    if s:
                        s["media"]["status"] = res.get("status","none")
                        s["media"]["best_image_url"] = res.get("best_image_url","")
                        s["media"]["candidates"] = res.get("candidates",[])
                        updated.append(slot_id)
                finally:
                    to_delete.append(job_id)
        for j in to_delete:
            del self.media_jobs[j]
        return updated

    def finalize(self) -> Dict[str, Any]:
        notes = []
        for sid in st.session_state.slot_order:
            s = self.slots.get(sid)
            if not s: continue
            if s.get("label") and s.get("bullets"):
                notes.append({"label": s["label"], "bullets": s["bullets"]})
        if len(notes) < 4:
            return {"ok": False, "error": "Need 4 filled slots"}
        final = finalize_card_with_gpt(notes)
        st.session_state.final_lines = final or {}
        return {"ok": bool(final)}

# =========================
# FLOW: on user message
# =========================
def process_user_message(text: str, orch: Orchestrator):
    # 1) Detect public entity (fast heuristic + GPT router)
    e_type, e_name = normalize_entity(text)
    if not e_type:
        e_type, e_name = route_entity_with_gpt(text)

    # If public entity: create/assign slot & schedule search
    if e_type and e_type != "none" and e_name:
        sid = current_slot_id()
        # set slot label early (for UI)
        label_hint = {
            "book":"Must-Read",
            "podcast":"Podcast",
            "person":"Role Model",
            "tool":"Go-to Tool",
            "film":"Influence"
        }.get(e_type, "Item")
        orch.upsert_slot(sid, label=f"{label_hint} — {e_name}", bullets=None, done=False)
        orch.schedule_media_search(sid, e_type, e_name)

    # 2) Distill insights from user's message into bullets (for current or next slot)
    sid2 = current_slot_id()
    hint = e_name or ""
    lab, bullets = distill_insights_with_gpt(text, context_hint=hint)
    if bullets:
        # If current slot already used for entity, prefer next slot for practice—unless empty.
        # Simple policy: if current has a media search pending and no bullets yet, fill it; else next slot.
        target_sid = sid2
        s_cur = st.session_state.slots.get(sid2)
        if s_cur and s_cur.get("media",{}).get("status") in ("searching","found","generated") and s_cur.get("bullets"):
            # current already has bullets -> move to next
            if st.session_state.current_slot_ix < 3:
                st.session_state.current_slot_ix += 1
                target_sid = current_slot_id()
        orch.upsert_slot(target_sid, label=lab or (hint if hint else "Practice"), bullets=bullets, done=True)
        # Advance after filling
        advance_slot()

    # 3) Generate next short question (always visible)
    reply = chat_reply(None)  # history includes user's turn; no extra text needed
    st.session_state.history.append({"role":"assistant","content": reply})

# =========================
# UI
# =========================
def render_progress_and_timeline():
    slots = st.session_state.slots
    filled = sum(1 for s in slots.values() if s.get("label") and s.get("bullets"))
    st.progress(min(1.0, filled/4), text=f"Progress: {filled}/4")

    if not slots: return
    cols = st.columns(4)
    for i, sid in enumerate(st.session_state.slot_order):
        s = slots.get(sid)
        with cols[i]:
            st.markdown(f"**{(s or {}).get('label') or sid}**")
            if s:
                st.caption("status: " + (s.get("media",{}).get("status") or "pending"))
                best = (s.get("media",{}).get("best_image_url") or "").strip()
                if best:
                    try: st.image(best, use_container_width=True)
                    except Exception: st.caption("(image unavailable)")
                else:
                    st.caption("(image pending)")
            else:
                st.caption("(empty)")

def render_chat_history():
    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def render_overrides():
    st.subheader("Media Overrides (optional)")
    for sid, s in st.session_state.slots.items():
        with st.expander(f"Override for {s.get('label') or sid}"):
            c1, c2 = st.columns(2)
            with c1:
                url = st.text_input("Image URL (http/https)", key=f"url_{sid}")
                if url and url.startswith("http"):
                    s["media"]["status"] = "found"
                    s["media"]["best_image_url"] = url
                    st.success("Using URL override.")
            with c2:
                up = st.file_uploader("Upload image", type=["png","jpg","jpeg"], key=f"up_{sid}")
                if up:
                    path = save_uploaded_image(up, sid)
                    if path:
                        s["media"]["status"] = "uploaded"
                        s["media"]["best_image_url"] = path
                        st.success("Using uploaded image.")

def render_final_card():
    lines = st.session_state.final_lines
    if not lines: return
    st.subheader("Your Expert Card")
    for sid in st.session_state.slot_order:
        s = st.session_state.slots.get(sid)
        if not s: continue
        label = s.get("label","").strip()
        if not label: continue
        line = lines.get(label,"").strip()
        if not line: continue
        best = (s.get("media",{}).get("best_image_url") or "").strip()
        cols = st.columns([1,2])
        with cols[0]:
            if best:
                try: st.image(best, use_container_width=True)
                except Exception: st.caption("(image unavailable)")
            else:
                st.caption("(image pending)")
        with cols[1]:
            st.markdown(f"**{label}**")
            st.write(line)

# =========================
# MAIN
# =========================
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
st.caption("GPT-5 only • Hosted Web Search • Async media • One short question per turn")

client = get_client()
init_state()
orch = Orchestrator()

# Poll media jobs every run
updated_slots = orch.poll_media()
if updated_slots:
    for sid in updated_slots:
        st.toast(f"Media updated: {sid}", icon="🖼️")

render_progress_and_timeline()
render_chat_history()

# Initial assistant question
if not any(m["role"] == "assistant" for m in st.session_state.history):
    st.session_state.history.append({
        "role":"assistant",
        "content":"Hi! Which single book most changed how you think about building data-driven products? Title only."
    })

# Chat input
user_text = st.chat_input("Your turn…")
if user_text:
    st.session_state.history.append({"role":"user","content": user_text})
    # Process immediately (entity routing, schedule media, insights, next question)
    process_user_message(user_text, orch)
    st.rerun()

# Action buttons
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("✨ Finalize (needs 4 slots)"):
        res = orch.finalize()
        if not res.get("ok"):
            st.warning(res.get("error","Could not finalize"))
        else:
            st.success("Finalized!")
            st.rerun()

with c2:
    if st.button("🔄 Restart"):
        try:
            st.session_state.executor.shutdown(cancel_futures=True)
        except Exception:
            pass
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

with c3:
    if st.button("🧹 Clear Final"):
        st.session_state.final_lines = {}
        st.rerun()

render_final_card()
render_overrides()
