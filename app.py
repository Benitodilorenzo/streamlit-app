# app_expert_card_gpt5.py
# Single-file Streamlit Demo (GPT-5, Async Media, Function Calls, Streaming)

import os, re, uuid, urllib.parse, requests
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

# =========================================================
# CONFIG
# =========================================================
APP_TITLE = "🟡 Expert Card Creator — GPT-5 (Async + Function Calls)"
os.environ.setdefault("OPENAI_CHAT_MODEL", "gpt-5")  # default model
STREAMING = True             # Streaming für flüssigen Chat
HTTP_TIMEOUT = 8             # Netz-Timeouts
MEDIA_MAX_WORKERS = 4        # parallele Such-Jobs
PLACEHOLDER_MIN_SIZE = 20000 # min. Bytes für Bild (HEAD)

# bevorzugte Domains je Typ (Ranking & Query-Shaping)
PREFER = {
    "book":    ["m.media-amazon.com","books.google.com","openlibrary.org","wikipedia.org"],
    "podcast": ["podcasts.apple.com","spotify.com","nytimes.com"],
    "person":  ["wikipedia.org","wikidata.org"],
    "tool":    ["wikipedia.org","docs.","developer."],
    "film":    ["themoviedb.org","wikipedia.org","imdb.com"]
}

MEDIA_DIR = os.path.join(os.getcwd(), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

# GPT-5 optionale Regler (standardmäßig AUS – wir stabilisieren NICHT)
REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT")  # "low"|"medium"|"high" oder None
VERBOSITY        = os.getenv("OPENAI_VERBOSITY")         # "low"|"default"|"high" oder None

# =========================================================
# OPENAI CLIENT
# =========================================================
def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    return OpenAI(api_key=key)

client: Optional[OpenAI] = None

# =========================================================
# UI / SESSION STATE
# =========================================================
def init_state():
    if "initialized" in st.session_state:
        return
    st.session_state.initialized = True
    st.session_state.history: List[Dict[str,Any]] = []
    st.session_state.slots: Dict[str, Dict[str,Any]] = {}  # S1..S4
    st.session_state.final_lines: Dict[str,str] = {}
    st.session_state.media_jobs: Dict[str, Tuple[str, Future]] = {}
    st.session_state.executor = ThreadPoolExecutor(max_workers=MEDIA_MAX_WORKERS)
    st.session_state.phase = "discovery"
    st.session_state.slot_order = ["S1","S2","S3","S4"]
    st.session_state.current_slot_ix = 0
    push_core_policy()

def push_core_policy():
    core_policy = (
        "ROLE\n"
        "You are the Interview & Orchestration agent. Your job is to craft a 4-item Expert Card.\n\n"
        "RULES\n"
        "- Ask exactly ONE short question per turn.\n"
        "- Do not ask for confirmations the user already gave; assume and proceed.\n"
        "- If the item is public (book, podcast, person, tool, film), call schedule_media_search.\n"
        "- If the item is conceptual or private (practice, principle, contrarian view), do NOT call media search.\n"
        "- When a slot is ready, call save_slot(slot_id, label, bullets = 2–4 concise facts).\n"
        "- Stop at 4 slots; then call finalize_card(notes).\n\n"
        "TONE\n"
        "Warm, specific, professional. No fluff."
    )
    st.session_state.history.append({"role":"system","content": core_policy})

def phase_prompt() -> str:
    ph = st.session_state.phase
    if ph == "discovery":
        return "GOAL (Discovery)\nElicit one high-signal item or theme (book, podcast, person, tool, film, practice, concept). Prefer concrete nouns (titles, names). Keep it narrow."
    if ph == "deepening":
        return "GOAL (Deepening)\nExtract impact or example: what changed, where it saved time/money/risk, or a vivid use case. If the item is public, capture precise title/person/brand; otherwise extract a concrete example."
    if ph == "consolidation":
        return "GOAL (Consolidation)\nChoose a precise label (e.g., Must-Read, Podcast, Role Model, Go-to Tool, Practice, Contrarian View). Craft 2–4 concise bullets (facts, evidence, phrasing cues). Save slot."
    return "GOAL (Finalize)\nThere are 4 slots. Produce 4 labeled lines (1–2 sentences each), specific and grounded in the notes. Call finalize_card(notes)."

def few_shots_for_phase() -> List[Dict[str,Any]]:
    ph = st.session_state.phase
    shots: List[Dict[str,Any]] = []
    if ph == "discovery":
        # Buch
        shots.append({"role":"user","content":"Data-Inspired von Sebastian Wernicke."})
        shots.append({
            "role":"assistant",
            "content":"Welcher Aspekt daraus hat deine Praxis am stärksten verändert?",
            "tool_calls":[{"name":"schedule_media_search","arguments":{
                "slot_id":"S1",
                "entity_type":"book",
                "entity_name":"Data-Inspired — Sebastian Wernicke",
                "prefer_domains":PREFER["book"]
            }}]
        })
        # Podcast
        shots.append({"role":"user","content":"Ich höre Hard Fork am meisten."})
        shots.append({
            "role":"assistant",
            "content":"Welche Folge würdest du Einsteiger*innen zuerst empfehlen – und warum?",
            "tool_calls":[{"name":"schedule_media_search","arguments":{
                "slot_id":"S1",
                "entity_type":"podcast",
                "entity_name":"Hard Fork — The New York Times",
                "prefer_domains":PREFER["podcast"]
            }}]
        })
    elif ph == "deepening":
        shots.append({"role":"user","content":"Ich bin geduldiger in AI-Trainings geworden."})
        shots.append({"role":"assistant","content":"An welcher Stelle im Training sparst du dadurch heute am meisten Friktion?"})
        shots.append({"role":"user","content":"Lieblingsfilm: Arrival."})
        shots.append({
            "role":"assistant",
            "content":"Was nimmst du aus Arrival mit, das sich in deiner Arbeit tatsächlich zeigt?",
            "tool_calls":[{"name":"schedule_media_search","arguments":{
                "slot_id":"S2",
                "entity_type":"film",
                "entity_name":"Arrival (2016)",
                "prefer_domains":PREFER["film"]
            }}]
        })
    elif ph == "consolidation":
        shots.append({
            "role":"assistant",
            "content":"Wen folgst du aktuell, um bei Datenstrategie frisch zu bleiben? Nur ein Name.",
            "tool_calls":[{"name":"save_slot","arguments":{
                "slot_id":"S1",
                "label":"Must-Read",
                "bullets":[
                    "Data-Inspired — Sebastian Wernicke",
                    "Daten leiten, Kontext entscheidet; keine Kennzahlen-Folklore",
                    "Kürzere Hypothesen-Loops in Workshops"
                ]
            }}]
        })
    else:  # finalize
        shots.append({
            "role":"assistant",
            "content":"Danke — ich habe genug Notizen.",
            "tool_calls":[{"name":"finalize_card","arguments":{
                "notes":[
                    {"label":"Must-Read","bullets":["..."]},
                    {"label":"Podcast","bullets":["..."]},
                    {"label":"Practice","bullets":["..."]},
                    {"label":"Contrarian View","bullets":["..."]}
                ]
            }}]
        })
    return shots

def current_slot_id() -> str:
    ix = st.session_state.current_slot_ix
    return st.session_state.slot_order[min(ix, 3)]

def mark_slot_done_and_advance():
    st.session_state.current_slot_ix = min(st.session_state.current_slot_ix + 1, 3)
    filled = len([s for s in st.session_state.slots.values() if s.get("done")])
    if filled >= 4:
        st.session_state.phase = "finalize"
    elif filled >= 2:
        st.session_state.phase = "consolidation"
    elif filled >= 1:
        st.session_state.phase = "deepening"
    else:
        st.session_state.phase = "discovery"

# =========================================================
# TOOLS (Function Calls)
# =========================================================
TOOLS = [
    {
        "name": "save_slot",
        "description": "Create or update a slot with a label and 2–4 concise bullets.",
        "parameters": {
            "type":"object",
            "properties":{
                "slot_id":{"type":"string","description":"S1..S4"},
                "label":{"type":"string"},
                "bullets":{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":4}
            },
            "required":["slot_id","label","bullets"]
        }
    },
    {
        "name": "schedule_media_search",
        "description": "Schedule async image search for a public item (book/podcast/person/tool/film). Returns a job_id immediately.",
        "parameters": {
            "type":"object",
            "properties":{
                "slot_id":{"type":"string"},
                "entity_type":{"type":"string","enum":["book","podcast","person","tool","film"]},
                "entity_name":{"type":"string","description":"Title + optional author/brand/year"},
                "prefer_domains":{"type":"array","items":{"type":"string"}}
            },
            "required":["slot_id","entity_type","entity_name"]
        }
    },
    {
        "name": "finalize_card",
        "description": "Produce the 4 final labeled lines (1–2 sentences each) from collected notes.",
        "parameters": {
            "type":"object",
            "properties":{
                "notes":{
                    "type":"array",
                    "items":{
                        "type":"object",
                        "properties":{
                            "label":{"type":"string"},
                            "bullets":{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":4}
                        },
                        "required":["label","bullets"]
                    },
                    "minItems":4,"maxItems":4
                }
            },
            "required":["notes"]
        }
    }
]

# =========================================================
# WEBSEARCH / IMAGE (in-file Agent)
# =========================================================
UA = "ExpertCardBot/1.0 (+https://example.local)"

def _get(url, params=None, headers=None, timeout=HTTP_TIMEOUT):
    h = {"User-Agent": UA}
    if headers: h.update(headers)
    return requests.get(url, params=params, headers=h, timeout=timeout)

def _head(url, timeout=HTTP_TIMEOUT):
    h = {"User-Agent": UA}
    return requests.head(url, headers=h, timeout=timeout, allow_redirects=True)

def build_query(entity_name: str, entity_type: str, prefer_domains: Optional[List[str]] = None) -> str:
    pd = prefer_domains or []
    dom = " OR ".join([f"site:{d}" for d in pd]) if pd else ""
    qname = entity_name.strip()
    if entity_type == "book":
        base = f'("{qname}") (cover|book)'
    elif entity_type == "podcast":
        base = f'("{qname}") podcast artwork'
    elif entity_type == "person":
        base = f'("{qname}") portrait'
    elif entity_type == "tool":
        base = f'("{qname}") logo|product'
    elif entity_type == "film":
        base = f'("{qname}") poster'
    else:
        base = f'"{qname}"'
    return f"{base} {dom}".strip()

def from_google_books(title_author: str) -> List[Dict[str, Any]]:
    try:
        r = _get("https://www.googleapis.com/books/v1/volumes", params={"q": title_author, "maxResults": 5})
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
                    out.append({
                        "image_url": url,
                        "page_url": info.get("infoLink") or "",
                        "title": info.get("title",""),
                        "source": "google_books"
                    })
                    break
        return out
    except Exception:
        return []

def from_openlibrary(q: str) -> List[Dict[str, Any]]:
    try:
        r = _get("https://openlibrary.org/search.json", params={"q": q, "limit": 5})
        if r.status_code != 200: return []
        data = r.json()
        out = []
        for d in data.get("docs", []):
            cid = d.get("cover_i")
            if not cid: continue
            img = f"https://covers.openlibrary.org/b/id/{cid}-L.jpg"
            page = f"https://openlibrary.org{d.get('key','')}"
            out.append({"image_url": img, "page_url": page, "title": d.get("title",""), "source": "openlibrary"})
        return out
    except Exception:
        return []

def from_wikipedia(q: str, langs: List[str] = ["en","de"]) -> List[Dict[str, Any]]:
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
                    outs.append({"image_url": img, "page_url": f"{base}/wiki/{urllib.parse.quote(title)}", "title": title, "source": "wikipedia"})
            if outs: break
        except Exception:
            continue
    return outs

def from_itunes_podcast(q: str) -> List[Dict[str, Any]]:
    try:
        r = _get("https://itunes.apple.com/search", params={"term": q, "media":"podcast","limit":5})
        if r.status_code != 200: return []
        data = r.json().get("results", [])
        out = []
        for d in data:
            img = d.get("artworkUrl600") or d.get("artworkUrl100") or d.get("artworkUrl60")
            page = d.get("collectionViewUrl") or d.get("trackViewUrl") or ""
            if img:
                out.append({"image_url": img, "page_url": page, "title": d.get("collectionName",""), "source": "itunes"})
        return out
    except Exception:
        return []

def from_google_cse_images(query: str) -> List[Dict[str, Any]]:
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
            imgobj = it.get("image") or {}
            w, h = imgobj.get("width", 0), imgobj.get("height", 0)
            ctx = it.get("image", {}).get("contextLink") or it.get("link")
            if link:
                out.append({"image_url": link, "page_url": ctx, "title": it.get("title",""), "width": w, "height": h, "source": "google_cse"})
        return out
    except Exception:
        return []

def is_http_image_like(url: str) -> bool:
    return isinstance(url,str) and url.startswith("http")

def tech_validate_and_dedup(cands: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    seen = set(); out = []
    for c in cands:
        u = c.get("image_url","")
        if not is_http_image_like(u): continue
        key = u.split("?")[0]
        if key in seen: continue
        try:
            h = _head(u, timeout=5)
            ctype = h.headers.get("Content-Type","")
            clen  = int(h.headers.get("Content-Length","0") or 0)
            if not (h.status_code < 400 and ctype.startswith("image/") and clen > PLACEHOLDER_MIN_SIZE):
                continue
        except Exception:
            continue
        seen.add(key)
        out.append(c)
    return out

def name_tokens_match(name: str, text: str) -> bool:
    if not name or not text: return False
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", name) if len(t)>2]
    t2 = text.lower()
    hit = sum(1 for t in tokens if t in t2)
    return hit >= max(1, len(tokens)//3)

def rank_candidates(entity_name: str, entity_type: str, cands: List[Dict[str,Any]], prefer_domains: Optional[List[str]]) -> List[Dict[str,Any]]:
    def score(c):
        s = 0
        u = c.get("image_url","").lower()
        for d in (prefer_domains or []):
            if d.lower() in u: s += 30
        if "m.media-amazon.com" in u: s += 20
        if "books.google.com"    in u: s += 12
        if "openlibrary.org"     in u or "covers.openlibrary.org" in u: s += 10
        if "wikipedia.org"       in u: s += 10
        if u.endswith((".jpg",".jpeg",".png",".webp")): s += 5
        w, h = c.get("width",0), c.get("height",0)
        s += min(int(min(w,h)/200), 10)
        if name_tokens_match(entity_name, c.get("title","")): s += 8
        return s
    return sorted(cands, key=score, reverse=True)

def vision_validate_if_needed(entity_name: str, entity_type: str, ranked: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    # Optionaler Hook: Vision-LLM für Top-2 (z. B. Titelerkennung). Standard: aus.
    return ranked

def resolve_media(entity_type: str, entity_name: str, prefer_domains: Optional[List[str]] = None) -> Dict[str,Any]:
    q = build_query(entity_name, entity_type, prefer_domains)
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
        cands += from_google_cse_images(q)

    cands = tech_validate_and_dedup(cands)
    ranked = rank_candidates(entity_name, entity_type, cands, prefer_domains)
    ranked = vision_validate_if_needed(entity_name, entity_type, ranked)

    if ranked:
        best = ranked[0]["image_url"]
        return {"status":"found","best_image_url":best,"candidates":ranked[:3],"reason":"ranked selection","confidence":0.85}
    else:
        return {"status":"none","best_image_url":"","candidates":[],"reason":"no suitable image","confidence":0.0}

# =========================================================
# PLACEHOLDER / UPLOAD
# =========================================================
def generate_placeholder_icon(text: str, slot_id: str) -> str:
    size = (640, 640)
    img = Image.new("RGB", size, color=(24, 31, 55))
    draw = ImageDraw.Draw(img)
    for r, col in [(260, (57, 96, 199)), (200, (73, 199, 142)), (140, (255, 205, 86))]:
        draw.ellipse([(size[0]//2-r, size[1]//2-r), (size[0]//2+r, size[1]//2+r)], outline=col, width=8)
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

def save_uploaded_image(file, slot_id: str) -> str:
    try:
        img = Image.open(file)
        path = os.path.join(MEDIA_DIR, f"{slot_id}_upload.png")
        img.save(path, format="PNG")
        return path
    except Exception:
        return ""

# =========================================================
# ORCHESTRATOR (Tools-Handler + Polling)
# =========================================================
class Orchestrator:
    def __init__(self):
        self.slots: Dict[str, Dict[str,Any]] = st.session_state.slots
        self.media_jobs: Dict[str, Tuple[str, Future]] = st.session_state.media_jobs
        self.executor: ThreadPoolExecutor = st.session_state.executor
        self.final_lines: Dict[str,str] = st.session_state.final_lines

    # Tool handlers
    def save_slot(self, slot_id: str, label: str, bullets: List[str]) -> Dict[str,Any]:
        s = self.slots.get(slot_id, {"slot_id": slot_id, "media": {"status":"pending","best_image_url":"","candidates":[]}})
        s["label"] = label
        s["bullets"] = bullets[:4]
        s["done"] = True
        self.slots[slot_id] = s
        mark_slot_done_and_advance()
        return {"ok": True}

    def schedule_media_search(self, slot_id: str, entity_type: str, entity_name: str, prefer_domains: Optional[List[str]] = None) -> Dict[str,Any]:
        if slot_id not in self.slots:
            self.slots[slot_id] = {"slot_id": slot_id, "label": "", "bullets": [], "done": False, "media": {"status":"searching","best_image_url":"","candidates":[]}}
        else:
            self.slots[slot_id].setdefault("media", {})["status"] = "searching"
        job_id = str(uuid.uuid4())[:8]
        fut = st.session_state.executor.submit(resolve_media, entity_type, entity_name, prefer_domains)
        self.media_jobs[job_id] = (slot_id, fut)
        return {"job_id": job_id}

    def finalize_card(self, notes: List[Dict[str,Any]]) -> Dict[str,Any]:
        # Minimaler Finalizer (du kannst hier gpt-5 responses nutzen)
        out = {}
        for n in notes[:4]:
            out[n["label"]] = " ".join(n["bullets"])[:220]
        st.session_state.final_lines = out
        return {"ok": True, "labels": list(out.keys())}

    # Polling
    def poll_media_jobs(self) -> List[str]:
        ready_slots = []
        to_delete = []
        for job_id, (slot_id, fut) in list(self.media_jobs.items()):
            if fut.done():
                try:
                    res = fut.result()
                    s = self.slots.get(slot_id)
                    if s:
                        if res.get("status") == "none":
                            path = generate_placeholder_icon(s.get("label") or slot_id, slot_id)
                            s["media"]["status"] = "generated"
                            s["media"]["best_image_url"] = path
                            s["media"]["candidates"] = []
                        else:
                            s["media"]["status"] = res.get("status","none")
                            s["media"]["best_image_url"] = res.get("best_image_url","")
                            s["media"]["candidates"] = res.get("candidates",[])
                    ready_slots.append(slot_id)
                finally:
                    to_delete.append(job_id)
        for job_id in to_delete:
            del self.media_jobs[job_id]
        return ready_slots

orchestrator = Orchestrator()

# =========================================================
# CHAT (Streaming + Function Calls)
# =========================================================
def build_messages(user_text: Optional[str]) -> List[Dict[str,Any]]:
    msgs = list(st.session_state.history)  # copy
    msgs.append({"role":"system","content": phase_prompt()})
    for ex in few_shots_for_phase():
        msgs.append(ex)
    if user_text:
        msgs.append({"role":"user","content": user_text})
    return msgs

def handle_tool_calls(tool_calls: List[Any]):
    for tc in tool_calls:
        name = tc.get("name") or tc.get("function",{}).get("name")
        args_str = tc.get("arguments") or tc.get("function",{}).get("arguments") or "{}"
        try:
            import json as _json
            args = args_str if isinstance(args_str, dict) else _json.loads(args_str)
        except Exception:
            args = {}
        if name == "save_slot":
            orchestrator.save_slot(args.get("slot_id","S1"), args.get("label",""), args.get("bullets",[]))
        elif name == "schedule_media_search":
            orchestrator.schedule_media_search(
                args.get("slot_id","S1"),
                args.get("entity_type","book"),
                args.get("entity_name",""),
                args.get("prefer_domains",[])
            )
        elif name == "finalize_card":
            orchestrator.finalize_card(args.get("notes",[]))

def _add_gpt5_overrides(kwargs: Dict[str,Any]) -> Dict[str,Any]:
    extra = {}
    if REASONING_EFFORT: extra["reasoning_effort"] = REASONING_EFFORT
    if VERBOSITY:        extra["verbosity"] = VERBOSITY
    if extra: kwargs.update(extra)
    return kwargs

def chat_once_stream(user_text: Optional[str]) -> str:
    msgs = build_messages(user_text)
    model = os.getenv("OPENAI_CHAT_MODEL","gpt-5")

    if STREAMING:
        container = st.chat_message("assistant")
        stream_area = container.empty()
        acc = ""
        try:
            kwargs = _add_gpt5_overrides({
                "model": model,
                "messages": msgs,
                "tools": TOOLS,
                "tool_choice": "auto"
            })
            with client.chat.completions.stream(**kwargs) as stream:
                for event in stream:
                    if event.type == "token":
                        acc += event.token
                        stream_area.markdown(acc)
                    elif event.type == "tool_call":
                        pass
                final = stream.get_final_response()
                msg = final.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None) or []
                handle_tool_calls([{
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                } for tc in tool_calls])
                return acc or (msg.content or "")
        except Exception:
            pass  # Fallback unten

    kwargs = _add_gpt5_overrides({
        "model": model,
        "messages": msgs,
        "tools": TOOLS,
        "tool_choice": "auto"
    })
    res = client.chat.completions.create(**kwargs)
    msg = res.choices[0].message
    content = msg.content or ""
    with st.chat_message("assistant"):
        st.markdown(content)
    tool_calls = getattr(msg, "tool_calls", None) or []
    handle_tool_calls([{
        "name": tc.function.name,
        "arguments": tc.function.arguments
    } for tc in tool_calls])
    return content

# =========================================================
# RENDERING
# =========================================================
def render_progress_and_timeline():
    slots = st.session_state.slots
    filled = len([s for s in slots.values() if s.get("label") and s.get("bullets")])
    st.progress(min(1.0, filled/4), text=f"Progress: {filled}/4")
    if not slots: return
    cols = st.columns(min(len(slots),4))
    for i, sid in enumerate(sorted(slots.keys())):
        s = slots[sid]
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
        if m["role"] == "system": continue
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def render_overrides():
    st.subheader("Media Overrides (optional)")
    for sid, s in st.session_state.slots.items():
        with st.expander(f"Override for {s.get('label') or sid}"):
            col1, col2 = st.columns(2)
            with col1:
                url = st.text_input("Image URL (http/https)", key=f"url_{sid}")
                if url and url.startswith("http"):
                    s["media"]["status"] = "found"
                    s["media"]["best_image_url"] = url
                    st.success("Using URL override.")
            with col2:
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
        label = s.get("label","")
        if not label: continue
        line = lines.get(label,"")
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

# =========================================================
# MAIN
# =========================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
st.caption("GPT-5 • Streaming Chat • Async Media Search (Books/OpenLibrary/Wikipedia/iTunes; Google CSE fallback)")

client = get_client()
init_state()

# Poll Media Jobs (nicht-blockierend)
ready = orchestrator.poll_media_jobs()
if ready:
    for sid in ready:
        st.toast(f"Media updated for {sid}", icon="🖼️")

# Progress + Timeline + History
render_progress_and_timeline()
render_chat_history()

# Erste Assistentenfrage, falls noch leer
if not any(m["role"] == "assistant" for m in st.session_state.history):
    with st.chat_message("assistant"):
        st.markdown("Hi! Welches einzelne Buch hat am stärksten beeinflusst, wie du datengetriebene Produkte denkst? Nur der Titel.")

# Chat Input
user_text = st.chat_input("Deine Antwort…")
if user_text:
    st.session_state.history.append({"role":"user","content": user_text})
    reply = chat_once_stream(None)  # user_text ist bereits in build_messages enthalten
    if reply:
        st.session_state.history.append({"role":"assistant","content": reply})
    st.rerun()

# Final Card
render_final_card()

# Overrides + Controls
render_overrides()
c1, c2 = st.columns(2)
with c1:
    if st.button("🔄 Restart"):
        try:
            st.session_state.executor.shutdown(cancel_futures=True)
        except Exception:
            pass
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
with c2:
    if st.button("🧹 Clear final"):
        st.session_state.final_lines = {}
        st.rerun()
