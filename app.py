# app_expert_card_gpt5.py
# Single-file Streamlit App — GPT-5 only, Hosted Web Search (Responses API),
# Async media, 3-Agent-Architektur (reines Prompting), dynamische Gesprächsführung.

# Anforderungen:
#   pip install streamlit openai pillow requests
# Env:
#   OPENAI_API_KEY                 (required)
#   OPENAI_GPT5_SNAPSHOT           (optional, default: gpt-5-2025-08-07)
#   OPENAI_CHAT_MODEL              (optional, default: snapshot)

import os, re, uuid, json, time, traceback
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
OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", MODEL_SNAPSHOT)  # wir bleiben bei GPT-5 only

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

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")  # wir akzeptieren auch webp großformatig

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

    # Erwartung für nächste Antwort (steuert Item-Erkennung — nur Bias)
    st.session_state.expected_type: Optional[str] = None  # "book" | "podcast" | "person" | "tool" | "film" | None

def current_slot_id() -> str:
    ix = st.session_state.current_slot_ix
    return st.session_state.slot_order[min(ix, 3)]

def advance_slot():
    st.session_state.current_slot_ix = min(st.session_state.current_slot_ix + 1, 3)

# =========================
# UTILS
# =========================
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

def deep_find_image_urls(obj: Any) -> List[str]:
    """Durchsuche beliebige verschachtelte Strukturen nach direkten Bild-URLs."""
    found = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v)
                # manchmal liegen URLs in Strings unter keys wie "url" / "image_url" etc.
                if isinstance(v, str):
                    s = v.strip()
                    if s.startswith("http") and any(s.lower().split("?")[0].endswith(ext) for ext in IMG_EXT):
                        found.append(s)
        elif isinstance(x, list):
            for it in x:
                walk(it)
        elif isinstance(x, str):
            s = x.strip()
            if s.startswith("http") and any(s.lower().split("?")[0].endswith(ext) for ext in IMG_EXT):
                found.append(s)
    walk(obj)
    # einfache Deduplizierung
    out, seen = [], set()
    for u in found:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

# =========================
# LLM HELPERS — Agent-2: Entity Router
# =========================
def normalize_entity(raw: str, expected_type: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Einfaches Heuristik-Bias; wir nutzen *zusätzlich* den LLM-Router."""
    s = (raw or "").strip()
    if not s:
        return (None, None)

    if expected_type in {"book", "podcast", "person", "tool", "film"}:
        name = s.strip(' "“”')
        m = re.search(r'["“]?([^"\n]+?)["”]?\s+(?:by|von|from)\s+([A-Za-zÀ-ÖØ-öø-ÿ\.\-\s]+)$', s, re.I)
        if expected_type == "book" and m:
            title = m.group(1).strip()
            return ("book", title)
        return (expected_type, name)

    # generische Buch-Heuristik (Titel by/von Autor)
    m = re.search(r'["“]?([^"\n]+?)["”]?\s+(?:by|von|from)\s+([A-Za-zÀ-ÖØ-öø-ÿ\.\-\s]+)$', s, re.I)
    if m:
        title = m.group(1).strip()
        return ("book", title)

    # kurzer Titel -> häufig Buch
    words = s.split()
    if 1 <= len(words) <= 6 and any(w[0:1].isupper() for w in words):
        return ("book", s.strip(' "“”'))

    return (None, None)

def route_entity_with_gpt(text: str, expected_type: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    sys = (
        "You extract ONE public entity if present. Types: book|podcast|person|tool|film|none.\n"
        "Return ONLY JSON: {\"detected\":true|false,\"entity_type\":\"book|podcast|person|tool|film|none\",\"entity_name\":\"...\"}.\n"
    )
    if expected_type:
        sys += f"Strong prior: The user was just asked for a {expected_type}. Prefer that type if plausible.\n"

    fewshot = [
        {"role":"user","content":"Data Inspired"},
        {"role":"assistant","content":'{"detected":true,"entity_type":"book","entity_name":"Data Inspired"}'},
        {"role":"user","content":"Lex Fridman podcast"},
        {"role":"assistant","content":'{"detected":true,"entity_type":"podcast","entity_name":"Lex Fridman Podcast"}'},
        {"role":"user","content":"Demis Hassabis"},
        {"role":"assistant","content":'{"detected":true,"entity_type":"person","entity_name":"Demis Hassabis"}'},
        {"role":"user","content":"The Neverending Story"},
        {"role":"assistant","content":'{"detected":true,"entity_type":"book","entity_name":"The Neverending Story"}'},
    ]
    try:
        msgs = [{"role":"system","content": sys}]
        msgs.extend(fewshot)
        msgs.append({"role":"user","content": text})
        resp = client.chat.completions.create(model=OPENAI_MODEL, messages=msgs)
        data = parse_json_loose(resp.choices[0].message.content or "")
        if data and data.get("detected") and data.get("entity_type") != "none":
            return (data.get("entity_type"), data.get("entity_name"))
    except Exception:
        pass
    return (None, None)

# =========================
# LLM HELPERS — Agent-2: Hosted Web Search
# =========================
def hosted_search_best_image(entity_type: str, entity_name: str) -> Dict[str, Any]:
    """
    Hosted Web Search via Responses API.
    WICHTIG:
      - Tool-Typ: web_search_preview
      - tool_choice NICHT erzwingen (auto/omit).
      - Wir parsen output_text *und* die Items (resp.model_dump()) auf Bild-URLs.
    """
    prefer = PREFERRED_DOMAINS.get(entity_type, [])
    instruction = (
        "Use web search to find ONE authoritative image for the item below.\n"
        f"- Item type: {entity_type}\n"
        f"- Item: {entity_name}\n"
        f"- Prefer domains: {', '.join(prefer) if prefer else 'none'}\n"
        "- Return a direct image URL (.jpg/.jpeg/.png/.webp), not an HTML page.\n"
        "- Avoid thumbnails/watermarks/memes/wrong items.\n"
        '- Output *only* JSON: {"url":"...","page_url":"...","source":"...","confidence":0-1,"reason":"..."}'
    )
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            tools=[{"type": "web_search_preview"}],  # << korrekt für GPT-5-Snapshot
            # tool_choice weglassen (auto) → Modell entscheidet; starke Instruktion zwingt praktisch die Nutzung
            input=[{
                "role":"user",
                "content":[{"type":"input_text","text": instruction}]
            }]
        )

        # 1) Versuche direktes JSON aus output_text
        out_text = getattr(resp, "output_text", None)
        data = parse_json_loose(out_text or "")

        # 2) Falls leer: gesamte Struktur inspizieren (Items können Tool-Ergebnisse tragen)
        if not data:
            try:
                raw = json.loads(resp.model_dump_json())
                # Sammle alle potenziellen Bild-URLs
                urls = deep_find_image_urls(raw)
                pick = urls[0] if urls else ""
                if pick:
                    # Rekonstruiere page_url/source heuristisch aus Struktur
                    source = ""
                    page_url = ""
                    # grobe Heuristik: suche neben dem URL-Feld Strings mit wikipedia/amazon/...
                    txt_blob = json.dumps(raw)
                    for dom in (prefer or []):
                        if dom in txt_blob:
                            source = dom
                            break
                    data = {"url": pick, "page_url": page_url or "", "source": source or "", "confidence": 0.55, "reason": "extracted from tool items"}
            except Exception:
                pass

        if isinstance(data, dict) and data.get("url"):
            return {"status":"found","best_image_url": data["url"], "candidates":[data], "notes": data.get("reason","")}
        return {"status":"error","best_image_url":"","candidates":[],"notes":"web_search produced no usable image JSON"}
    except Exception as e:
        return {"status":"error","best_image_url":"","candidates":[],"notes": f"web_search error: {e}"}

def preflight_web_search():
    """Einfacher Probecall; kein erzwungenes tool_choice; prüft Verfügbarkeit & meldet Klartextfehler."""
    try:
        _ = client.responses.create(
            model=OPENAI_MODEL,
            tools=[{"type":"web_search_preview"}],
            input=[{"role":"user","content":[{"type":"input_text","text":"Return JSON: {\"url\":\"https://example.com/a.jpg\",\"page_url\":\"https://example.com\",\"source\":\"example.com\",\"confidence\":0.1,\"reason\":\"test\"}"}]}]
        )
        st.session_state.web_search_ok = True
    except Exception as e:
        st.session_state.web_search_ok = False
        st.session_state.web_search_err = str(e)

# =========================
# LLM HELPERS — Agent-1 (Interview) & Agent-3 (Finalize)
# =========================
def agent1_next_question(conversation: List[Dict[str,str]], slot_summaries: List[str]) -> str:
    """
    Chat-Agent (LLM) generiert *eine* passende, kurze nächste Frage.
    Few-Shots decken: a) typisches Data/AI-Kontext-Buch, b) untypisches Buch (Fantasy),
    c) Podcast, d) Person, e) Ausweichantworten.
    """
    sys = (
        "ROLE You are a concise, warm interviewer crafting a 4-item Expert Card.\n"
        "GOALS (in parallel):\n"
        "  1) Surface PUBLIC items (book, podcast, person, tool, film) naturally.\n"
        "  2) Elicit the user's personal angle/practice on them.\n"
        "RULES\n"
        "  - Exactly ONE short question per turn.\n"
        "  - React to what the user actually said; acknowledge atypical picks.\n"
        "  - If user gave only a title/name, ask for their angle/connection (not generic facts).\n"
        "  - If user gives rich context, probe for a concrete practice, example, or decision.\n"
        "  - Keep it human and specific. No filler.\n"
    )
    fewshot = [
        {"role":"user","content":"Data-Inspired by Sebastian Wernicke."},
        {"role":"assistant","content":"What did you change in how you run experiments after reading it?"},
        {"role":"user","content":"Honestly, The Neverending Story by Michael Ende."},
        {"role":"assistant","content":"Love that. What’s the bridge between that story and how you make decisions at work?"},
        {"role":"user","content":"I follow Demis Hassabis a lot."},
        {"role":"assistant","content":"What concrete practice of his have you actually adopted day-to-day?"},
        {"role":"user","content":"I mainly listen to Hard Fork."},
        {"role":"assistant","content":"Which single episode would you recommend first — and why?"},
        {"role":"user","content":"Not sure. I just like good stories."},
        {"role":"assistant","content":"What’s one story that stuck — and what did it change in your approach?"}
    ]
    msgs = [{"role":"system","content": sys}]
    msgs.extend(conversation[-8:])  # nur letzte Züge für Frische
    msgs.extend(fewshot)
    msgs.append({"role":"user","content":"(Give exactly one short follow-up question next.)"})
    try:
        resp = client.chat.completions.create(model=OPENAI_MODEL, messages=msgs, temperature=0.7)
        q = (resp.choices[0].message.content or "").strip()
        # Sicherheitskürzung
        if "\n" in q: q = q.split("\n",1)[0].strip()
        if len(q) > 200: q = q[:200].rstrip(" .") + "?"
        return q or "What changed in your practice after this?"
    except Exception:
        return "What changed in your practice after this?"

def agent3_finalize(notes: List[Dict[str, Any]]) -> Dict[str, str]:
    prompt = (
        "Create a concise Expert Card with exactly 4 labeled items.\n"
        "Each line: **Label — one crisp sentence** (grounded in bullets; no fluff). Return plain text."
    )
    try:
        txt = "\n".join(f"- {n['label']}: " + "; ".join(n.get("bullets", [])) for n in notes)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content": prompt},{"role":"user","content": txt}],
            temperature=0.3
        )
        out = (resp.choices[0].message.content or "").strip()
        lines = [l.strip("- ").strip() for l in out.splitlines() if l.strip()]
        res: Dict[str,str] = {}
        for l in lines:
            if "—" in l:
                lab, body = l.split("—",1)
            elif ":" in l:
                lab, body = l.split(":",1)
            else:
                continue
            res[lab.strip()] = body.strip()
        return res
    except Exception:
        return {}

def distill_insights_if_rich(text: str, context_hint: str = "") -> Tuple[Optional[str], List[str]]:
    # Nur wenn Substanz (mehr als Titel) → Einsichten extrahieren
    if len(text.split()) < 6:
        return (None, [])
    sys = (
        "From the user's message, extract one concise 'label' and 2–4 crisp bullets of their personal angle/practice. "
        "Avoid general facts. Return ONLY JSON: {\"label\":\"...\",\"bullets\":[\"...\",\"...\"]}"
    )
    if context_hint:
        sys += f" Context hint: {context_hint}"
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content": sys},{"role":"user","content": text}],
            temperature=0.2
        )
        data = parse_json_loose(resp.choices[0].message.content or "")
        if data and data.get("label") and isinstance(data.get("bullets"), list):
            return (data["label"], [b for b in data["bullets"] if isinstance(b,str)][:4])
    except Exception:
        pass
    return (None, [])

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
        if label is not None: s["label"] = label[:160]
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
        try:
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
        except Exception as e:
            return {
                "slot_id": slot_id,
                "status":"error",
                "best_image_url":"",
                "candidates":[],
                "notes": f"job error: {e}"
            }

    def poll_media(self) -> List[str]:
        updated, to_delete = [], []
        for job_id, (slot_id, fut) in list(self.media_jobs.items()):
            if fut.done():
                res = {"status":"error","best_image_url":"","candidates":[],"notes":"unknown"}
                try:
                    res = fut.result()
                except Exception as e:
                    res["notes"] = f"future error: {e}"
                s = self.slots.get(slot_id)
                if s:
                    m = s["media"]
                    m["status"] = res.get("status","error")
                    m["best_image_url"] = res.get("best_image_url","")
                    m["candidates"] = res.get("candidates",[])
                    m["notes"] = res.get("notes","")
                    updated.append(slot_id)
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
        final = agent3_finalize(notes)
        st.session_state.final_lines = final or {}
        return {"ok": bool(final)}

# =========================
# FLOW
# =========================
def process_user_message(text: str, orch: Orchestrator):
    # Agent-2: Item-Erkennung (nur dann Websuche)
    e_type, e_name = normalize_entity(text, expected_type=st.session_state.expected_type)
    if not e_type:
        e_type, e_name = route_entity_with_gpt(text, expected_type=st.session_state.expected_type)

    if e_type and e_type != "none" and e_name:
        sid = current_slot_id()
        label_hint = {
            "book":"Must-Read", "podcast":"Podcast", "person":"Role Model",
            "tool":"Go-to Tool", "film":"Influence"
        }.get(e_type, "Item")
        orch.upsert_slot(sid, label=f"{label_hint} — {e_name}", bullets=None, done=False)
        orch.schedule_media_search(sid, e_type, e_name)

    # Agent-1: Einsichten (nur bei substanzieller Antwort)
    sid2 = current_slot_id()
    lab, bullets = distill_insights_if_rich(text, context_hint=e_name or "")
    if bullets:
        orch.upsert_slot(sid2, label=lab or (e_name or "Practice"), bullets=bullets, done=True)
        advance_slot()

    # Agent-1: Nächste Frage LLM-geführt
    q = agent1_next_question(st.session_state.history + [{"role":"user","content": text}], [])
    st.session_state.history.append({"role":"assistant","content": q})
    # Erwartung loslassen (oder situativ setzen)
    st.session_state.expected_type = None

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
                up = st.file_uploader("Upload image", type=["png","jpg","jpeg","webp"], key=f"up_{sid}")
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

# =========================
# MAIN
# =========================
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
st.caption("GPT-5 only • Hosted Web Search • Async media • One short question per turn")

client = get_client()
init_state()

# Preflight Hosted Web Search (einmalig)
if st.session_state.web_search_ok is None:
    preflight_web_search()
if st.session_state.web_search_ok is False:
    st.error(f"Hosted web_search not available: {st.session_state.web_search_err}")

orch = Orchestrator()

# Seed-Frage + Erwartung
if not any(m["role"] == "assistant" for m in st.session_state.history):
    st.session_state.history.append({
        "role":"assistant",
        "content":"Hi! Which single book most changed how you think about building data-driven products? Title only."
    })
    st.session_state.expected_type = "book"

# Media-Jobs poll’en
updated_slots = orch.poll_media()
if updated_slots:
    for sid in updated_slots:
        st.toast(f"Media updated: {sid}", icon="🖼️")

# Render
render_progress_and_timeline()
render_chat_history()

# Input
user_text = st.chat_input("Your turn…")
if user_text:
    st.session_state.history.append({"role":"user","content": user_text})
    process_user_message(user_text, orch)
    st.rerun()

# Actions
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
