# app_three_agents_minimal.py
# Streamlit Single-File Demo — GPT-5 Chat Agent + Websearch Agent + Finalizer
# - Chat-Agent: nur Interview + Routing (save_slot / schedule_media_search)
# - Websearch-Agent: echte Websuche (Books/Wikipedia/iTunes + Google CSE Fallback), HEAD-Validierung
# - Finalizer: baut 4 Linien aus Slots
# - Async Media via ThreadPoolExecutor
# - Kein "Nachdenken"-Flag, keine Fake-Fewshots, minimaler Prompt, kurze Latenz

import os, re, uuid, urllib.parse, requests
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

# =========================
# CONFIG
# =========================
APP_TITLE = "🟡 Expert Card — 3 Agents (Chat · Search · Finalize)"
OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5")  # GPT-5
HTTP_TIMEOUT = 8
MEDIA_MAX_WORKERS = 4
MIN_IMAGE_BYTES = 20000  # HEAD Content-Length

PREFER = {
    "book":    ["m.media-amazon.com","books.google.com","openlibrary.org","wikipedia.org"],
    "podcast": ["podcasts.apple.com","spotify.com"],
    "person":  ["wikipedia.org","wikidata.org"],
    "tool":    ["wikipedia.org","docs.","developer."],
    "film":    ["themoviedb.org","wikipedia.org","imdb.com"],
}

MEDIA_DIR = os.path.join(os.getcwd(), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

UA = "ExpertCard/1.0 (+https://local.demo)"


# =========================
# OPENAI CLIENT
# =========================
def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    return OpenAI(api_key=key)

client: Optional[OpenAI] = None


# =========================
# SESSION / STATE
# =========================
def init_state():
    if "initialized" in st.session_state:
        return
    st.session_state.initialized = True

    # Gesprächsverlauf (nur realer Dialog; Policies bauen wir pro Turn frisch)
    st.session_state.history: List[Dict[str,Any]] = []

    # SLOTS: S1..S4
    st.session_state.slots: Dict[str, Dict[str,Any]] = {}
    st.session_state.slot_order = ["S1","S2","S3","S4"]
    st.session_state.current_slot_ix = 0

    # Media-Jobs
    st.session_state.executor = ThreadPoolExecutor(max_workers=MEDIA_MAX_WORKERS)
    st.session_state.media_jobs: Dict[str, Tuple[str, Future]] = {}

    # Final Lines
    st.session_state.final_lines: Dict[str,str] = {}

def current_slot_id() -> str:
    ix = st.session_state.current_slot_ix
    return st.session_state.slot_order[min(ix, 3)]

def mark_slot_done_and_advance():
    st.session_state.current_slot_ix = min(st.session_state.current_slot_ix + 1, 3)


# =========================
# CHAT AGENT — prompt + tools
# =========================
CORE_POLICY = (
    "ROLE\n"
    "You are a concise interviewer. Your goals:\n"
    "1) Identify PUBLIC items (book, podcast, person, tool, film) — call schedule_media_search immediately.\n"
    "2) Capture the user's personal insight/practice about those items — call save_slot with a label and 2–4 bullets.\n\n"
    "RULES\n"
    "- Exactly ONE short question per turn.\n"
    "- If the user provided a PUBLIC item, do not ask for confirmation — call schedule_media_search in the same turn.\n"
    "- If the item is private/conceptual (e.g., Practice, Principle), do NOT call schedule_media_search.\n"
    "- Stop at 4 slots.\n"
    "- Keep the tone warm, specific, professional; no fluff.\n\n"
    "ENTITY TYPES\n"
    "book | podcast | person | tool | film\n"
)

def state_summary() -> str:
    # kompaktes State-Update für das Modell
    parts = []
    for sid in st.session_state.slot_order:
        s = st.session_state.slots.get(sid)
        if not s:
            parts.append(f"{sid}: (empty)")
            continue
        label = s.get("label","")
        done  = "done" if s.get("done") else "open"
        parts.append(f"{sid}: {label or '(pending)'} [{done}]")
    return "STATE\n" + " | ".join(parts) + f"\nCURRENT_SLOT_ID={current_slot_id()}"

# Tools (Function Calls) — NUR zwei: save_slot, schedule_media_search
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_slot",
            "description": "Save a slot with label and 2–4 concise bullets (facts/insights).",
            "parameters": {
                "type": "object",
                "properties": {
                    "slot_id": {"type": "string", "description": "S1..S4"},
                    "label": {"type": "string"},
                    "bullets": {
                        "type": "array",
                        "items": {"type":"string"},
                        "minItems": 2,
                        "maxItems": 4
                    }
                },
                "required": ["slot_id","label","bullets"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_media_search",
            "description": "For a PUBLIC item (book/podcast/person/tool/film), schedule image search now. entity_name should be just the human-readable string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slot_id": {"type":"string"},
                    "entity_type": {"type":"string","enum":["book","podcast","person","tool","film"]},
                    "entity_name": {"type":"string"},
                },
                "required": ["slot_id","entity_type","entity_name"]
            }
        }
    }
]

def build_messages(user_text: Optional[str]) -> List[Dict[str,Any]]:
    msgs: List[Dict[str,Any]] = []
    # Minimaler Verlauf: erste system policy + letzter Dialogtail
    msgs.append({"role":"system","content": CORE_POLICY})
    msgs.append({"role":"system","content": state_summary()})

    # letzter Tail (max 6 turns)
    tail = [m for m in st.session_state.history if m["role"] in ("user","assistant")]
    tail = tail[-6:]
    msgs.extend(tail)

    if user_text:
        msgs.append({"role":"user","content": user_text})
    return msgs


# =========================
# ORCHESTRATOR — Toolhandler + Finalizer + Media-Poll
# =========================
class Orchestrator:
    def __init__(self):
        self.slots = st.session_state.slots
        self.media_jobs = st.session_state.media_jobs
        self.executor: ThreadPoolExecutor = st.session_state.executor

    # save_slot — vom Chat-Agent gerufen
    def save_slot(self, slot_id: str, label: str, bullets: List[str]) -> Dict[str,Any]:
        s = self.slots.get(slot_id, {"slot_id": slot_id, "media":{"status":"pending","best_image_url":"","candidates":[]}})
        s["label"] = label.strip()[:80]
        s["bullets"] = [b.strip() for b in bullets][:4]
        s["done"] = True
        self.slots[slot_id] = s
        mark_slot_done_and_advance()
        return {"ok": True}

    # schedule_media_search — vom Chat-Agent gerufen
    def schedule_media_search(self, slot_id: str, entity_type: str, entity_name: str) -> Dict[str,Any]:
        s = self.slots.get(slot_id, {"slot_id": slot_id, "label":"", "bullets":[], "done": False, "media":{}})
        s["media"] = {"status":"searching","best_image_url":"","candidates":[]}
        self.slots[slot_id] = s
        job_id = str(uuid.uuid4())[:8]
        fut = self.executor.submit(resolve_media, entity_type, entity_name, PREFER.get(entity_type, []))
        self.media_jobs[job_id] = (slot_id, fut)
        return {"job_id": job_id}

    # Finalizer — baue 4 Linien (separater Agent, hier ein einfacher Prompt)
    def finalize(self) -> Dict[str,Any]:
        notes = []
        for sid in st.session_state.slot_order:
            s = self.slots.get(sid)
            if not s or not s.get("label") or not s.get("bullets"): continue
            notes.append({"label": s["label"], "bullets": s["bullets"]})
        if len(notes) < 4:
            return {"ok": False, "error":"Need 4 slots"}
        prompt = (
            "You turn notes into a concise Expert Card. Return exactly 4 items.\n"
            "Format:\n"
            "- Label: one or two short sentences, grounded in bullets; no fluff.\n"
        )
        msgs = [{"role":"system","content": prompt},
                {"role":"user","content": "\n".join(f"- {n['label']}: " + "; ".join(n['bullets']) for n in notes)}]
        res = client.chat.completions.create(model=OPENAI_MODEL, messages=msgs)
        txt = (res.choices[0].message.content or "").strip()
        lines = [l.strip("- ").strip() for l in txt.splitlines() if l.strip()]
        out: Dict[str,str] = {}
        for l in lines:
            if ":" in l:
                lab, body = l.split(":",1)
                out[lab.strip()] = body.strip()
        st.session_state.final_lines = out
        return {"ok": True, "labels": list(out.keys())}

    # Media-Poll — Ergebnisse einsammeln
    def poll_media(self) -> List[str]:
        updated = []
        to_del = []
        for job_id, (slot_id, fut) in list(self.media_jobs.items()):
            if fut.done():
                try:
                    res = fut.result()
                    s = self.slots.get(slot_id)
                    if s:
                        if res.get("status") == "found":
                            s["media"]["status"] = "found"
                            s["media"]["best_image_url"] = res.get("best_image_url","")
                            s["media"]["candidates"] = res.get("candidates",[])
                        else:
                            # Placeholder
                            path = generate_placeholder_icon(s.get("label") or slot_id, slot_id)
                            s["media"]["status"] = "generated"
                            s["media"]["best_image_url"] = path
                            s["media"]["candidates"] = []
                        updated.append(slot_id)
                finally:
                    to_del.append(job_id)
        for j in to_del:
            del self.media_jobs[j]
        return updated


# =========================
# WEBSEARCH AGENT (Backend)
# =========================
def _get(url, params=None, headers=None, timeout=HTTP_TIMEOUT):
    h = {"User-Agent": UA}
    if headers: h.update(headers)
    return requests.get(url, params=params, headers=h, timeout=timeout)

def _head(url, timeout=HTTP_TIMEOUT):
    h = {"User-Agent": UA}
    return requests.head(url, headers=h, timeout=timeout, allow_redirects=True)

def from_google_books(q: str) -> List[Dict[str,Any]]:
    try:
        r = _get("https://www.googleapis.com/books/v1/volumes", params={"q": q, "maxResults": 5})
        if r.status_code != 200: return []
        data = r.json()
        out = []
        for it in data.get("items", []):
            info = it.get("volumeInfo", {})
            links = info.get("imageLinks", {}) or {}
            for key in ["extraLarge","large","medium","small","thumbnail","smallThumbnail"]:
                url = links.get(key)
                if url:
                    if url.startswith("http://"): url = "https://" + url[7:]
                    out.append({"image_url": url, "page_url": info.get("infoLink") or "", "title": info.get("title",""), "source":"google_books"})
                    break
        return out
    except Exception:
        return []

def from_openlibrary(q: str) -> List[Dict[str,Any]]:
    try:
        r = _get("https://openlibrary.org/search.json", params={"q": q, "limit":5})
        if r.status_code != 200: return []
        data = r.json()
        out = []
        for d in data.get("docs", []):
            cid = d.get("cover_i")
            if not cid: continue
            img = f"https://covers.openlibrary.org/b/id/{cid}-L.jpg"
            page = f"https://openlibrary.org{d.get('key','')}"
            out.append({"image_url": img, "page_url": page, "title": d.get("title",""), "source":"openlibrary"})
        return out
    except Exception:
        return []

def from_wikipedia(q: str, langs: List[str] = ["en","de"]) -> List[Dict[str,Any]]:
    outs = []
    for lang in langs:
        try:
            base = f"https://{lang}.wikipedia.org"
            s = _get(f"{base}/w/rest.php/v1/search/title", params={"q": q, "limit": 3})
            if s.status_code != 200: continue
            pages = s.json().get("pages", [])
            for p in pages:
                title = p.get("title")
                if not title: continue
                summ = _get(f"{base}/api/rest_v1/page/summary/{urllib.parse.quote(title)}")
                if summ.status_code != 200: continue
                sj = summ.json()
                img = (sj.get("originalimage") or {}).get("source") or (sj.get("thumbnail") or {}).get("source")
                if img:
                    outs.append({"image_url": img, "page_url": f"{base}/wiki/{urllib.parse.quote(title)}", "title": title, "source":"wikipedia"})
            if outs: break
        except Exception:
            continue
    return outs

def from_itunes_podcast(q: str) -> List[Dict[str,Any]]:
    try:
        r = _get("https://itunes.apple.com/search", params={"term": q, "media":"podcast","limit":5})
        if r.status_code != 200: return []
        res = r.json().get("results", [])
        out = []
        for d in res:
            img = d.get("artworkUrl600") or d.get("artworkUrl100") or d.get("artworkUrl60")
            page = d.get("collectionViewUrl") or d.get("trackViewUrl") or ""
            if img:
                out.append({"image_url": img, "page_url": page, "title": d.get("collectionName",""), "source":"itunes"})
        return out
    except Exception:
        return []

def from_google_cse_images(query: str) -> List[Dict[str,Any]]:
    cx = os.getenv("GOOGLE_CSE_CX","")
    key = os.getenv("GOOGLE_CSE_KEY","")
    if not cx or not key: return []
    try:
        r = _get("https://www.googleapis.com/customsearch/v1",
                params={"cx":cx,"key":key,"q":query,"searchType":"image","num":5,"safe":"active"})
        if r.status_code != 200: return []
        items = r.json().get("items", []) or []
        out = []
        for it in items:
            link = it.get("link")
            w = (it.get("image") or {}).get("width", 0)
            h = (it.get("image") or {}).get("height", 0)
            ctx = (it.get("image") or {}).get("contextLink") or it.get("link")
            if link:
                out.append({"image_url": link, "page_url": ctx, "title": it.get("title",""), "width": w, "height": h, "source":"google_cse"})
        return out
    except Exception:
        return []

def tech_validate(cands: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out = []
    seen = set()
    for c in cands:
        u = c.get("image_url","")
        if not (isinstance(u,str) and u.startswith("http")): continue
        key = u.split("?")[0]
        if key in seen: continue
        try:
            h = _head(u, timeout=5)
            ct = h.headers.get("Content-Type","")
            cl = int(h.headers.get("Content-Length","0") or 0)
            if not (h.status_code < 400 and ct.startswith("image/") and cl > MIN_IMAGE_BYTES):
                continue
        except Exception:
            continue
        seen.add(key)
        out.append(c)
    return out

def rank(entity_name: str, entity_type: str, cands: List[Dict[str,Any]], prefer: List[str]) -> List[Dict[str,Any]]:
    def score(c):
        s = 0
        u = (c.get("image_url","") or "").lower()
        t = (c.get("title","") or "")
        if prefer:
            for d in prefer:
                if d.lower() in u:
                    s += 30
        if "m.media-amazon.com" in u: s += 20
        if "books.google.com"    in u: s += 12
        if "openlibrary.org"     in u or "covers.openlibrary.org" in u: s += 10
        if "wikipedia.org"       in u: s += 10
        if u.endswith((".jpg",".jpeg",".png",".webp")): s += 5
        if entity_name and t and any(tok in t.lower() for tok in re.findall(r"[a-z0-9]{4,}", entity_name.lower())):
            s += 6
        return s
    return sorted(cands, key=score, reverse=True)

def build_query(entity_name: str, entity_type: str, prefer: List[str]) -> str:
    dom = " OR ".join([f"site:{d}" for d in prefer]) if prefer else ""
    if entity_type == "book":
        base = f'("{entity_name}") (cover|book)'
    elif entity_type == "podcast":
        base = f'("{entity_name}") podcast artwork'
    elif entity_type == "person":
        base = f'("{entity_name}") portrait'
    elif entity_type == "tool":
        base = f'("{entity_name}") logo'
    elif entity_type == "film":
        base = f'("{entity_name}") poster'
    else:
        base = f'"{entity_name}"'
    return f"{base} {dom}".strip()

def resolve_media(entity_type: str, entity_name: str, prefer: List[str]) -> Dict[str,Any]:
    cands: List[Dict[str,Any]] = []
    try:
        if entity_type == "book":
            cands += from_google_books(entity_name)
            if len(cands) < 2:
                cands += from_openlibrary(entity_name)
        elif entity_type == "podcast":
            cands += from_itunes_podcast(entity_name)
        elif entity_type in ("person","tool","film"):
            cands += from_wikipedia(entity_name)
    except Exception:
        pass
    if len(cands) < 2:
        q = build_query(entity_name, entity_type, prefer)
        cands += from_google_cse_images(q)
    cands = tech_validate(cands)
    ranked = rank(entity_name, entity_type, cands, prefer)
    if ranked:
        return {"status":"found","best_image_url": ranked[0]["image_url"], "candidates": ranked[:3]}
    return {"status":"none","best_image_url":"","candidates":[]}


# =========================
# PLACEHOLDER / UPLOAD
# =========================
def generate_placeholder_icon(text: str, slot_id: str) -> str:
    size = (640, 640)
    img = Image.new("RGB", size, color=(24,31,55))
    draw = ImageDraw.Draw(img)
    for r, col in [(260,(57,96,199)), (200,(73,199,142)), (140,(255,205,86))]:
        draw.ellipse([(size[0]//2-r, size[1]//2-r),(size[0]//2+r, size[1]//2+r)], outline=col, width=8)
    txt = (text or "Idea").strip()[:28]
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
    except Exception:
        font = ImageFont.load_default()
    w, h = draw.textsize(txt, font=font)
    draw.text(((size[0]-w)//2,(size[1]-h)//2), txt, fill=(240,240,240), font=font)
    path = os.path.join(MEDIA_DIR, f"{slot_id}_placeholder.png")
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


# =========================
# CHAT TURN (non-streaming; schnell & stabil)
# =========================
def handle_tool_calls(tool_calls: List[Any], orch: Orchestrator):
    import json as _json
    for tc in tool_calls or []:
        name = tc.function.name
        args = _json.loads(tc.function.arguments or "{}")
        if name == "save_slot":
            orch.save_slot(args["slot_id"], args["label"], args["bullets"])
        elif name == "schedule_media_search":
            orch.schedule_media_search(args["slot_id"], args["entity_type"], args["entity_name"])

def chat_once(user_text: Optional[str], orch: Orchestrator) -> str:
    msgs = build_messages(user_text)
    res = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs,
        tools=TOOLS,
        tool_choice="auto",
    )
    msg = res.choices[0].message
    content = msg.content or ""
    # Tool-Calls ausführen
    tcs = getattr(msg, "tool_calls", []) or []
    handle_tool_calls(tcs, orch)
    return content


# =========================
# UI RENDERING
# =========================
def render_progress_and_timeline():
    slots = st.session_state.slots
    filled = sum(1 for s in slots.values() if s.get("label") and s.get("bullets"))
    st.progress(min(1.0, filled/4), text=f"Progress: {filled}/4")

    if not slots: return
    cols = st.columns(min(4, max(1, len(slots))))
    for i, sid in enumerate(st.session_state.slot_order):
        s = slots.get(sid)
        if not s: continue
        with cols[i % len(cols)]:
            st.markdown(f"**{s.get('label') or sid}**")
            st.caption("status: " + (s.get("media",{}).get("status") or "pending"))
            best = (s.get("media",{}).get("best_image_url") or "").strip()
            if best:
                try: st.image(best, use_container_width=True)
                except Exception: st.caption("(image unavailable)")
            else:
                st.caption("(image pending)")

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
st.caption("Chat-Agent ↔ Websearch-Agent ↔ Finalizer • Parallel Media Search • Minimal Prompting")

client = get_client()
init_state()
orch = Orchestrator()

# Poll Media-Jobs bei jedem Run
updated = orch.poll_media()
if updated:
    for sid in updated:
        st.toast(f"Media updated for {sid}", icon="🖼️")

render_progress_and_timeline()
render_chat_history()

# Erste Frage, wenn leer
if not any(m["role"] == "assistant" for m in st.session_state.history):
    st.session_state.history.append({"role":"assistant","content":"Hi! Which single book most changed how you think about building data-driven products? Title only."})

# Chat Input
user_text = st.chat_input("Your turn…")
if user_text:
    st.session_state.history.append({"role":"user","content": user_text})
    reply = chat_once(None, orch)  # user turn liegt bereits in history
    if reply:
        st.session_state.history.append({"role":"assistant","content": reply})
    st.rerun()

# Finalizer Button
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("✨ Finalize Card (4 slots)"):
        res = orch.finalize()
        if not res.get("ok"):
            st.warning(res.get("error","Could not finalize"))
        else:
            st.success("Finalized!")
            st.rerun()
with c2:
    if st.button("🔄 Restart"):
        try: st.session_state.executor.shutdown(cancel_futures=True)
        except Exception: pass
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
with c3:
    if st.button("🧹 Clear Final"):
        st.session_state.final_lines = {}
        st.rerun()

render_final_card()
render_overrides()
