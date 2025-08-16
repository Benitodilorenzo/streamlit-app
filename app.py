# app_expert_card_gpt5.py
# Single-file Streamlit app — GPT-5 only, Hosted Web Search (Responses API),
# async media, minimal prompting — NO response_format / NO reasoning params.
#
# Requirements:
#   pip install streamlit openai pillow requests
#
# Environment:
#   OPENAI_API_KEY                (required)
#   OPENAI_GPT5_SNAPSHOT          (optional, default: gpt-5-2025-08-07)
#   OPENAI_CHAT_MODEL             (optional, default: snapshot)

import os, re, uuid, json, urllib.parse
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
MODEL_SNAPSHOT = os.getenv("OPENAI_GPT5_SNAPSHOT", "gpt-5-2025-08-07")
OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", MODEL_SNAPSHOT)

HTTP_TIMEOUT = 10
MEDIA_MAX_WORKERS = 4
PLACEHOLDER_MIN_TEXT = 3

MEDIA_DIR = os.path.join(os.getcwd(), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)
UA = "ExpertCard/1.0 (+https://local.app) Streamlit"

PREFERRED_DOMAINS = {
    "book":    ["m.media-amazon.com", "amazon.de", "books.google.com", "openlibrary.org", "wikipedia.org"],
    "podcast": ["podcasts.apple.com", "itunes.apple.com", "spotify.com"],
    "person":  ["wikipedia.org", "wikidata.org"],
    "tool":    ["wikipedia.org", "docs.", "developer."],
    "film":    ["wikipedia.org", "themoviedb.org", "imdb.com"],
}

# =========================
# CLIENT
# =========================
def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    return OpenAI(api_key=key)

client: Optional[OpenAI] = None

# =========================
# STATE
# =========================
def init_state():
    if st.session_state.get("initialized"):
        return
    st.session_state.initialized = True

    st.session_state.history: List[Dict[str, Any]] = []
    st.session_state.slots: Dict[str, Dict[str, Any]] = {}
    st.session_state.slot_order = ["S1", "S2", "S3", "S4"]
    st.session_state.current_slot_ix = 0

    st.session_state.media_fingerprints: set = set()
    st.session_state.executor = ThreadPoolExecutor(max_workers=MEDIA_MAX_WORKERS)
    st.session_state.media_jobs: Dict[str, Tuple[str, Future]] = {}

    st.session_state.final_lines: Dict[str, str] = {}
    st.session_state.web_search_ok = None
    st.session_state.web_search_err = ""

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
    if len(txt) < PLACEHOLDER_MIN_TEXT:
        txt = "Idea"
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
    except Exception:
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), txt, font=font)  # Pillow ≥10
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        w, h = draw.textsize(txt, font=font)          # Fallback
    draw.text(((size[0]-w)//2, (size[1]-h)//2), txt, fill=(240, 240, 240), font=font)

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
    s = (raw or "").strip()
    if not s: return (None, None)
    m = re.search(r'["“]?([^"\n]+?)["”]?\s+(?:by|von|from)\s+([A-Za-zÀ-ÖØ-öø-ÿ\.\-\s]+)$', s, re.I)
    if m:
        title = m.group(1).strip()
        author = m.group(2).strip()
        return ("book", f"{title} — {author}")
    if re.search(r'\bmovie|film|poster\b', s, re.I):
        return ("film", s.replace("poster", "").strip(' "'))
    if len(s.split()) in (2, 3) and s[0].isupper():
        return (None, None)
    return (None, None)

def parse_json_loose(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end+1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None

# =========================
# GPT-5 HELPERS (Hosted Web Search)
# =========================
def hosted_search_best_image(entity_type: str, entity_name: str) -> Dict[str, Any]:
    """
    Hosted Web Search via Responses API — NO response_format / NO reasoning.
    Model instructed to return JSON only:
      { "url": "...jpg|png", "page_url":"...", "source":"...", "confidence":0-1, "reason":"..." }
    """
    prefer = PREFERRED_DOMAINS.get(entity_type, [])
    instruction = (
        "Task: Find ONE authoritative image for the given public item.\n"
        f"- Item type: {entity_type}\n"
        f"- Prefer domains: {', '.join(prefer) if prefer else 'none'}\n"
        "- MUST return a direct image URL (.jpg or .png), not an HTML page.\n"
        "- Avoid thumbnails, memes, watermarks, or wrong items.\n"
        "- OUTPUT: a single JSON object with keys: url, page_url, source, confidence, reason. No extra text."
    )
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {"type": "input_text", "text": f"Item: {entity_name}"}
                    ]
                }
            ]
        )
        out_text = getattr(resp, "output_text", None)
        if not out_text and hasattr(resp, "output"):
            try:
                out_text = resp.output if isinstance(resp.output, str) else json.dumps(resp.output)
            except Exception:
                out_text = None
        data = parse_json_loose(out_text or "")
        if isinstance(data, dict) and data.get("url") and data.get("page_url"):
            return {"status":"found","best_image_url": data["url"], "candidates":[data], "notes": data.get("reason","")}
        return {"status":"error","best_image_url":"","candidates":[],"notes":"web_search returned no parseable JSON"}
    except Exception as e:
        return {"status":"error","best_image_url":"","candidates":[],"notes": f"web_search error: {e}"}

def preflight_web_search():
    """Probe hosted web_search using only input_text blocks."""
    try:
        _ = client.responses.create(
            model=OPENAI_MODEL,
            tools=[{"type":"web_search"}],
            tool_choice="auto",
            input=[{
                "role":"user",
                "content":[
                    {"type":"input_text","text":"Return JSON: {\"url\":\"https://example.com/a.jpg\",\"page_url\":\"https://example.com\",\"source\":\"example.com\",\"confidence\":0.1,\"reason\":\"test\"}"}
                ]
            }]
        )
        st.session_state.web_search_ok = True
    except Exception as e:
        st.session_state.web_search_ok = False
        st.session_state.web_search_err = str(e)

# =========================
# GPT-5 HELPERS (Chat: Routing / Insights / Finalize)
# =========================
def route_entity_with_gpt(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Chat Completions — ask for JSON ONLY; parse loosely."""
    sys = (
        "Extract one PUBLIC entity if present. Types: book|podcast|person|tool|film|none.\n"
        "If found, return ONLY JSON: {\"detected\":true,\"entity_type\":\"book|...\",\"entity_name\":\"...\"}\n"
        "Else return: {\"detected\":false,\"entity_type\":\"none\",\"entity_name\":\"\"}"
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content": sys},{"role":"user","content": text}]
        )
        data = parse_json_loose(resp.choices[0].message.content or "")
        if data and data.get("detected") and data.get("entity_type") != "none":
            return (data.get("entity_type"), data.get("entity_name"))
    except Exception:
        pass
    return (None, None)

def distill_insights_with_gpt(text: str, context_hint: str = "") -> Tuple[Optional[str], List[str]]:
    """Chat Completions — extract label + 2–4 bullets; return JSON-only & parse."""
    sys = (
        "From the user's message, extract one concise 'label' and 2–4 crisp 'bullets'. "
        "Return ONLY JSON: {\"label\":\"...\",\"bullets\":[\"...\",\"...\"]}"
    )
    if context_hint:
        sys += f" Context hint: {context_hint}"
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content": sys},{"role":"user","content": text}]
        )
        data = parse_json_loose(resp.choices[0].message.content or "")
        if data and data.get("label") and isinstance(data.get("bullets"), list):
            return (data["label"], [b for b in data["bullets"] if isinstance(b,str)][:4])
    except Exception:
        pass
    return (None, [])

def finalize_card_with_gpt(notes: List[Dict[str, Any]]) -> Dict[str, str]:
    prompt = (
        "Create a concise Expert Card with exactly 4 labeled items.\n"
        "Each item: 1–2 short sentences, grounded in bullets; no fluff.\n"
        "Return plain text with lines like: '- Label: text'"
    )
    try:
        txt = "\n".join(f"- {n['label']}: " + "; ".join(n.get("bullets", [])) for n in notes)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content": prompt},{"role":"user","content": txt}]
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
# CHAT AGENT
# =========================
CHAT_POLICY = (
    "ROLE\nYou are a concise interviewer.\n"
    "GOALS\n1) Identify PUBLIC items (book, podcast, person, tool, film).\n"
    "2) Get the user's personal insight/practice about them.\n"
    "RULES\n- Exactly ONE short question per turn.\n- If a PUBLIC item was mentioned, do not ask for confirmation.\n"
    "TONE\nWarm, specific, professional; no fluff."
)

def state_summary() -> str:
    parts = []
    for sid in st.session_state.slot_order:
        s = st.session_state.slots.get(sid)
        if not s:
            parts.append(f"{sid}: (empty)")
        else:
            done = "done" if s.get("done") else "open"
            parts.append(f"{sid}: {s.get('label','(pending)')} [{done}]")
    return "STATE\n" + " | ".join(parts) + f"\nCURRENT_SLOT_ID={current_slot_id()}"

def chat_reply() -> str:
    msgs = [
        {"role":"system","content": CHAT_POLICY},
        {"role":"system","content": state_summary()}
    ]
    tail = [m for m in st.session_state.history if m["role"] in ("user","assistant")]
    msgs.extend(tail[-6:])
    res = client.chat.completions.create(model=OPENAI_MODEL, messages=msgs)
    text = (res.choices[0].message.content or "").strip()
    if not text or not text.endswith("?"):
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
            "media": {"status":"pending","best_image_url":"","candidates":[],"notes":""}
        })
        if label is not None: s["label"] = label[:120]
        if bullets is not None: s["bullets"] = [b.strip() for b in bullets][:4]
        if done is not None: s["done"] = bool(done)
        self.slots[slot_id] = s

    def schedule_media_search(self, slot_id: str, entity_type: str, entity_name: str):
        key = f"{entity_type}|{entity_name}".lower().strip()
        if key in self.fp:
            return
        self.fp.add(key)

        s = self.slots.get(slot_id) or {
            "slot_id": slot_id, "label":"", "bullets":[], "done": False,
            "media": {"status":"pending","best_image_url":"","candidates":[],"notes":""}
        }
        s["media"] = {"status":"searching","best_image_url":"","candidates":[],"notes":""}
        self.slots[slot_id] = s

        fut = self.executor.submit(self._media_job, slot_id, entity_type, entity_name)
        job_id = str(uuid.uuid4())[:8]
        self.media_jobs[job_id] = (slot_id, fut)
        st.toast("🔎 Image search started…", icon="🖼️")

    def _media_job(self, slot_id: str, entity_type: str, entity_name: str) -> Dict[str, Any]:
        res = hosted_search_best_image(entity_type, entity_name)
        if res.get("status") in ("found","error"):
            res["slot_id"] = slot_id
            return res
        return {
            "slot_id": slot_id,
            "status":"generated",
            "best_image_url": generate_placeholder_icon(entity_name or entity_type, slot_id),
            "candidates": [],
            "notes": "placeholder"
        }

    def poll_media(self) -> List[str]:
        updated = []
        to_delete = []
        for job_id, (slot_id, fut) in list(self.media_jobs.items()):
            if fut.done():
                try:
                    res = fut.result()
                    s = self.slots.get(slot_id)
                    if s:
                        m = s["media"]
                        m["status"] = res.get("status","none")
                        m["best_image_url"] = res.get("best_image_url","")
                        m["candidates"] = res.get("candidates",[])
                        m["notes"] = res.get("notes","")
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
# FLOW
# =========================
def process_user_message(text: str, orch: Orchestrator):
    # 1) Entity detect (heuristic + GPT)
    e_type, e_name = normalize_entity(text)
    if not e_type:
        e_type, e_name = route_entity_with_gpt(text)

    # If public entity → schedule media
    if e_type and e_type != "none" and e_name:
        sid = current_slot_id()
        label_hint = {
            "book":"Must-Read", "podcast":"Podcast", "person":"Role Model",
            "tool":"Go-to Tool", "film":"Influence"
        }.get(e_type, "Item")
        orch.upsert_slot(sid, label=f"{label_hint} — {e_name}", bullets=None, done=False)
        orch.schedule_media_search(sid, e_type, e_name)

    # 2) Insights
    sid2 = current_slot_id()
    hint = e_name or ""
    lab, bullets = distill_insights_with_gpt(text, context_hint=hint)
    if bullets:
        target_sid = sid2
        s_cur = st.session_state.slots.get(sid2)
        if s_cur and s_cur.get("media",{}).get("status") in ("searching","found","generated") and s_cur.get("bullets"):
            if st.session_state.current_slot_ix < 3:
                st.session_state.current_slot_ix += 1
                target_sid = current_slot_id()
        orch.upsert_slot(target_sid, label=lab or (hint if hint else "Practice"), bullets=bullets, done=True)
        advance_slot()

    # 3) Next question
    reply = chat_reply()
    st.session_state.history.append({"role":"assistant","content": reply})

# =========================
# UI
# =========================
def render_progress_and_timeline():
    slots = st.session_state.slots
    filled = sum(1 for s in slots.values() if s.get("label") and s.get("bullets"))
    st.progress(min(1.0, filled/4), text=f"Progress: {filled}/4")

    cols = st.columns(4)
    for i, sid in enumerate(st.session_state.slot_order):
        s = slots.get(sid)
        with cols[i]:
            st.markdown(f"**{(s or {}).get('label') or sid}**")
            if not s:
                st.caption("(empty)")
                continue
            m = s.get("media",{})
            st.caption(f"status: {m.get('status') or 'pending'}")
            best = (m.get("best_image_url") or "").strip()
            if best:
                try: st.image(best, use_container_width=True)
                except Exception: st.caption("(image unavailable)")
            else:
                st.caption("(image pending)")
            notes = (m.get("notes") or "").strip()
            if notes:
                st.code(notes, language="text")

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
                    s["media"]["notes"] = "override url"
                    st.success("Using URL override.")
            with c2:
                up = st.file_uploader("Upload image", type=["png","jpg","jpeg"], key=f"up_{sid}")
                if up:
                    path = save_uploaded_image(up, sid)
                    if path:
                        s["media"]["status"] = "uploaded"
                        s["media"]["best_image_url"] = path
                        s["media"]["notes"] = "uploaded file"
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

# --- MAIN (Patch) ---

st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
st.caption("GPT-5 only • Hosted Web Search • Async media • One short question per turn")

client = get_client()
init_state()

# 1) Preflight hosted web_search once (vor UI-Render, aber nach State-Init)
if st.session_state.web_search_ok is None:
    preflight_web_search()

if st.session_state.web_search_ok is False:
    st.error(f"Hosted web_search not available: {st.session_state.web_search_err}")

orch = Orchestrator()

# 2) **Seed**: Erste Assistentenfrage früh setzen (VOR jeglichem Render!)
def ensure_seed_question():
    # Falls noch kein Assistant-Post im Verlauf ist, jetzt hinzufügen
    if not any(m["role"] == "assistant" for m in st.session_state.history):
        st.session_state.history.append({
            "role": "assistant",
            "content": "Hi! Which single book most changed how you think about building data-driven products? Title only."
        })

ensure_seed_question()

# 3) Poll Media-Jobs (kann Toasts erzeugen)
updated_slots = orch.poll_media()
if updated_slots:
    for sid in updated_slots:
        st.toast(f"Media updated: {sid}", icon="🖼️")

# 4) Jetzt ERST rendern (Timeline/Chat)
render_progress_and_timeline()
render_chat_history()

# 5) Eingabe
user_text = st.chat_input("Your turn…")

if user_text:
    # Alles stateful erledigen VOR UI-Render
    st.session_state.history.append({"role":"user","content": user_text})
    process_user_message(user_text, orch)
    # Sofort rerun, damit die eben hinzugefügte Assistant-Nachricht sichtbar wird
    st.rerun()

# 6) Actions & Final Card (rendern zum Schluss)
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
