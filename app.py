# app_expert_card_gpt5.py
# Single-file Streamlit App — GPT-5 only, Hosted Web Search (Responses API)
# 3-Agents (pure prompting), async media, dynamische Gesprächsführung

# pip install streamlit openai pillow requests
# ENV:
#   OPENAI_API_KEY               (required)
#   OPENAI_GPT5_SNAPSHOT         (optional, default: gpt-5-2025-08-07)
#   OPENAI_CHAT_MODEL            (optional, default: snapshot)

import os, json, re, uuid, time, random
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

# ========= Config =========
APP_TITLE = "🟡 Expert Card — GPT-5 (Hosted Web Search, Async, Single File)"
MODEL_SNAPSHOT = os.getenv("OPENAI_GPT5_SNAPSHOT", "gpt-5-2025-08-07")
OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", MODEL_SNAPSHOT)  # GPT-5 snapshot

HTTP_TIMEOUT = 12
MEDIA_MAX_WORKERS = 4
MEDIA_DIR = os.path.join(os.getcwd(), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

PREFERRED_DOMAINS = {
    "book":    ["m.media-amazon.com", "amazon.de", "books.google.com", "openlibrary.org", "wikipedia.org"],
    "podcast": ["podcasts.apple.com", "itunes.apple.com", "spotify.com"],
    "person":  ["wikipedia.org", "wikidata.org"],
    "tool":    ["wikipedia.org", "docs.", "developer."],
    "film":    ["wikipedia.org", "themoviedb.org", "imdb.com"],
}

# ========= OpenAI Client =========
def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    return OpenAI(api_key=key)

client: Optional[OpenAI] = None

# ========= State =========
def init_state():
    if st.session_state.get("init"):
        return
    st.session_state.init = True

    st.session_state.history: List[Dict[str,str]] = []      # chat transcript (role, content)
    st.session_state.slots: Dict[str,Dict[str,Any]] = {}    # S1..S4
    st.session_state.slot_order = ["S1","S2","S3","S4"]
    st.session_state.next_slot_ix = 0

    st.session_state.media_jobs: Dict[str, Tuple[str, Future]] = {}
    st.session_state.executor = ThreadPoolExecutor(max_workers=MEDIA_MAX_WORKERS)

    st.session_state.web_ok = None
    st.session_state.web_err = ""

    st.session_state.final_lines: Dict[str,str] = {}

def current_slot_id() -> str:
    return st.session_state.slot_order[min(st.session_state.next_slot_ix, 3)]

def advance_slot():
    st.session_state.next_slot_ix = min(st.session_state.next_slot_ix + 1, 3)

# ========= Utils =========
def draw_placeholder(text: str, slot_id: str) -> str:
    size=(640,640)
    img=Image.new("RGB", size, (24,31,55))
    d=ImageDraw.Draw(img)
    for r,c in [(260,(57,96,199)),(200,(73,199,142)),(140,(255,205,86))]:
        d.ellipse([(size[0]//2-r,size[1]//2-r),(size[0]//2+r,size[1]//2+r)], outline=c, width=8)
    try:
        font=ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
    except:
        font=ImageFont.load_default()
    txt=(text or "Idea").strip()[:22] or "Idea"
    try:
        bbox = d.textbbox((0,0), txt, font=font); w=bbox[2]-bbox[0]; h=bbox[3]-bbox[1]
    except:
        w,h = d.textsize(txt, font=font)
    d.text(((size[0]-w)//2,(size[1]-h)//2), txt, fill=(240,240,240), font=font)
    path=os.path.join(MEDIA_DIR, f"{slot_id}_ph.png"); img.save(path,"PNG"); return path

def parse_json_loose(s: str) -> Optional[dict]:
    if not s: return None
    try: return json.loads(s)
    except: pass
    a=s.find("{"); b=s.rfind("}")
    if a!=-1 and b!=-1 and b>a:
        try: return json.loads(s[a:b+1])
        except: return None
    return None

def deep_find_urls(obj: Any) -> List[str]:
    out, seen = [], set()
    def walk(x):
        if isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
        elif isinstance(x, str):
            s=x.strip()
            if s.startswith("http") and any(s.lower().split("?")[0].endswith(ext) for ext in IMG_EXT):
                if s not in seen:
                    seen.add(s); out.append(s)
    walk(obj)
    return out

# ========= Agent 1 — Interview (LLM-only) =========
def agent1_opening_question() -> str:
    sys = (
        "ROLE You are a concise, warm interviewer crafting a 4-item Expert Card.\n"
        "GOALS 1) Surface PUBLIC items (book/podcast/person/tool/film) naturally. "
        "2) Elicit the user's personal angle/practice. "
        "RULES: Exactly ONE short question, human, specific; vary phrasing; no filler."
    )
    few = [
        {"role":"user","content":"(start)"},
        {"role":"assistant","content":"What’s one book that genuinely shifted how you approach building with data or AI?"},
        {"role":"user","content":"(start again)"},
        {"role":"assistant","content":"Pick one influence that changed your data/AI practice — a book, podcast, person, or film?"}
    ]
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":sys}, *few, {"role":"user","content":"(start afresh)"}],
            temperature=0.95
        )
        q=(r.choices[0].message.content or "").strip()
        if "\n" in q: q=q.split("\n",1)[0].strip()
        return q or "What’s one influence that changed your approach to data/AI?"
    except:
        return "What’s one influence that changed your approach to data/AI?"

def agent1_next_question(transcript: List[Dict[str,str]]) -> str:
    sys = (
        "ROLE You are a concise, warm interviewer crafting a 4-item Expert Card.\n"
        "GOALS: surface PUBLIC items; elicit personal practice. "
        "If user mentioned an unusual/fiction item, ask for the bridge to work. "
        "RULES: Exactly ONE short question; react to what they said; no generic filler."
    )
    few = [
        {"role":"user","content":"The Neverending Story by Michael Ende."},
        {"role":"assistant","content":"Love that. What’s the thread you pulled from that story into your day-to-day decisions?"},
        {"role":"user","content":"I follow Demis Hassabis."},
        {"role":"assistant","content":"What specific habit of his have you actually adopted?"}
    ]
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":sys}, *transcript[-8:], *few, {"role":"user","content":"(Give one short follow-up only.)"}],
            temperature=0.85
        )
        q=(r.choices[0].message.content or "").strip()
        if "\n" in q: q=q.split("\n",1)[0].strip()
        return q or "What’s the bridge into your work?"
    except:
        return "What’s the bridge into your work?"

# ========= Agent 2 — Watcher + Hosted Web Search (single pass) =========
def agent2_watch_and_search(slot_id: str, conversation: List[Dict[str,str]]) -> Dict[str,Any]:
    """
    Eine Responses-Runde:
      - Identifiziere EIN öffentliches Item in der *letzten* User-Antwort (book/podcast/person/tool/film) – falls vorhanden.
      - Nur dann: nutze web_search_preview, finde *ein* autoritatives Bild (direkte Bild-URL).
      - Gib JSON zurück:
          {"action":"search","slot_id":slot_id,"entity_type":"book|...","entity_name":"...",
           "url":"...","page_url":"...","source":"...","confidence":0–1,"reason":"..."}
        oder {"action":"none"}.
    """
    # Baue schlanken Gesprächskontext (letzte User-Nachricht reicht meist)
    last_user = ""
    for m in reversed(conversation):
        if m["role"] == "user":
            last_user = m["content"]
            break

    instruction = (
        "You are a monitoring agent.\n"
        f"Next slot id: {slot_id}\n"
        "Task:\n"
        "1) From the LAST user message ONLY, extract ONE public item if present (types: book|podcast|person|tool|film).\n"
        "2) If none → return exactly: {\"action\":\"none\"}.\n"
        "3) If found → CALL web search to retrieve ONE authoritative image (direct .jpg/.jpeg/.png/.webp). "
        "Prefer domains if relevant. Validate it matches the item (no memes/watermarks/wrong entity).\n"
        "4) Then return exactly this JSON:\n"
        "{\"action\":\"search\",\"slot_id\":\"S?\",\"entity_type\":\"...\",\"entity_name\":\"...\","
        "\"url\":\"...\",\"page_url\":\"...\",\"source\":\"...\",\"confidence\":0.0-1.0,\"reason\":\"...\"}"
    )

    prefer = ", ".join(PREFERRED_DOMAINS.get("book","") + PREFERRED_DOMAINS.get("podcast","") + PREFERRED_DOMAINS.get("person","") + PREFERRED_DOMAINS.get("tool","") + PREFERRED_DOMAINS.get("film",""))

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            tools=[{"type":"web_search_preview"}],
            input=[{
                "role":"user",
                "content":[
                    {"type":"input_text","text": instruction},
                    {"type":"input_text","text": f"Prefer domains: {prefer}"},
                    {"type":"input_text","text": f"LAST USER MESSAGE:\n{last_user}"}
                ]
            }]
        )

        # 1) Direktes JSON
        txt = getattr(resp, "output_text", "") or ""
        data = parse_json_loose(txt)

        # 2) Falls kein JSON oder keine URL, versuche URLs aus Items zu ziehen
        if not data or (isinstance(data, dict) and data.get("action")=="search" and not data.get("url")):
            raw = json.loads(resp.model_dump_json())
            urls = deep_find_urls(raw)
            if urls:
                # minimalistischer Fallback, falls Modell keine Struktur baut
                if not data or not isinstance(data, dict):
                    data = {"action":"search","slot_id":slot_id,"entity_type":"book","entity_name":"","url":urls[0],
                            "page_url":"","source":"","confidence":0.55,"reason":"extracted from tool items"}
                elif data.get("action")=="search" and not data.get("url"):
                    data["url"]=urls[0]

        # 3) Validierung
        if isinstance(data, dict):
            if data.get("action")=="search" and data.get("url"):
                data.setdefault("slot_id", slot_id)
                return {"ok":True, "data":data, "response_id": getattr(resp,"id", None)}
            if data.get("action")=="none":
                return {"ok":True, "data":data, "response_id": getattr(resp,"id", None)}

        return {"ok":False, "error":"no usable JSON", "response_id": getattr(resp,"id", None)}
    except Exception as e:
        return {"ok":False, "error": f"{e}"}

# ========= Agent 3 — Finalize =========
def agent3_finalize(notes: List[Dict[str, Any]]) -> Dict[str,str]:
    sys = (
        "Create a concise Expert Card with exactly 4 labeled items.\n"
        "Each line: **Label — one crisp sentence** grounded in bullets. No filler."
    )
    src = "\n".join(f"- {n['label']}: " + "; ".join(n.get("bullets", [])) for n in notes)
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":src}],
            temperature=0.35
        )
        out=(r.choices[0].message.content or "").strip()
        res={}
        for line in out.splitlines():
            line=line.strip("- ").strip()
            if not line: continue
            if "—" in line: lab,body=line.split("—",1)
            elif ":" in line: lab,body=line.split(":",1)
            else: continue
            res[lab.strip()]=body.strip()
        return res
    except:
        return {}

# ========= Orchestrator & Flow =========
class Orchestrator:
    def __init__(self):
        self.slots = st.session_state.slots
        self.executor: ThreadPoolExecutor = st.session_state.executor
        self.media_jobs: Dict[str, Tuple[str, Future]] = {}

    def upsert_slot(self, slot_id: str, label: Optional[str]=None,
                    bullets: Optional[List[str]]=None, done: Optional[bool]=None):
        s = self.slots.get(slot_id, {
            "slot_id": slot_id,
            "label": "",
            "bullets": [],
            "done": False,
            "media": {"status":"pending","best_image_url":"","candidates":[],"notes":""},
        })
        if label is not None: s["label"]=label[:160]
        if bullets is not None: s["bullets"]=[b.strip() for b in bullets][:4]
        if done is not None: s["done"]=bool(done)
        self.slots[slot_id]=s

    def schedule_agent2_job(self, slot_id: str, transcript: List[Dict[str,str]]):
        fut = self.executor.submit(agent2_watch_and_search, slot_id, transcript)
        jid = str(uuid.uuid4())[:8]
        self.media_jobs[jid]=(slot_id, fut)
        st.toast("🔎 Watching & searching…", icon="🖼️")

    def poll_media(self) -> List[str]:
        updated=[]
        for jid,(slot_id,fut) in list(self.media_jobs.items()):
            if fut.done():
                try:
                    res=fut.result()
                    s=self.slots.get(slot_id)
                    if not s: 
                        del self.media_jobs[jid]; continue
                    m=s["media"]
                    if res.get("ok") and isinstance(res.get("data"), dict):
                        data=res["data"]
                        if data.get("action")=="search" and data.get("url"):
                            m["status"]="found"
                            m["best_image_url"]=data["url"]
                            m["candidates"]=[data]
                            m["notes"]=data.get("reason","")
                        elif data.get("action")=="none":
                            m["status"]="none"
                            m["notes"]="no public item in last message"
                        else:
                            m["status"]="error"; m["notes"]="unusable agent2 JSON"
                    else:
                        m["status"]="error"; m["notes"]=res.get("error","agent2 error")
                    updated.append(slot_id)
                except Exception as e:
                    s=self.slots.get(slot_id)
                    if s:
                        s["media"]["status"]="error"
                        s["media"]["notes"]=f"future error: {e}"
                    updated.append(slot_id)
                finally:
                    del self.media_jobs[jid]
        return updated

    def finalize(self) -> Dict[str,Any]:
        notes=[]
        for sid in st.session_state.slot_order:
            s=self.slots.get(sid)
            if s and s.get("label") and s.get("bullets"):
                notes.append({"label": s["label"], "bullets": s["bullets"]})
        if len(notes)<4:
            return {"ok":False, "error":"Need 4 filled slots"}
        st.session_state.final_lines = agent3_finalize(notes)
        return {"ok": bool(st.session_state.final_lines)}

def process_user_turn(text: str, orch: Orchestrator):
    # 1) Agent 2 wird IMMER aufgerufen (watcher). Er entscheidet selbst "none" vs. "search".
    sid = current_slot_id()
    transcript = st.session_state.history + [{"role":"user","content":text}]
    # vorab Slot anlegen (nur in UI sichtbar) – Label bleibt leer bis Einsichten da sind
    if sid not in orch.slots:
        orch.upsert_slot(sid, label="", bullets=[], done=False)
    orch.schedule_agent2_job(sid, transcript)

    # 2) Einsichten NUR wenn Inhalt > bloßer Titel
    if len(text.split()) >= 6:
        sys = (
            "Extract a short label and 2–4 bullets reflecting the user's personal angle/practice; "
            "avoid general facts. Return ONLY JSON {\"label\":\"...\",\"bullets\":[\"...\",...]}"
        )
        try:
            r = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":sys},{"role":"user","content":text}],
                temperature=0.25
            )
            data = parse_json_loose(r.choices[0].message.content or "")
            if data and data.get("label"):
                orch.upsert_slot(sid, label=data["label"], bullets=data.get("bullets",[]), done=True)
                advance_slot()
        except:
            pass

    # 3) Agent 1 generiert die nächste Frage
    q = agent1_next_question(transcript)
    st.session_state.history.append({"role":"assistant","content": q})

# ========= UI =========
def render_timeline():
    slots=st.session_state.slots
    filled=sum(1 for s in slots.values() if s.get("label") and s.get("bullets"))
    st.progress(min(1.0, filled/4), text=f"Progress: {filled}/4")
    cols=st.columns(4)
    for i,sid in enumerate(st.session_state.slot_order):
        s=slots.get(sid)
        with cols[i]:
            st.markdown(f"**{(s or {}).get('label') or sid}**")
            if not s:
                st.caption("(empty)"); continue
            m=s.get("media",{})
            st.caption(f"status: {m.get('status','pending')}")
            url=(m.get("best_image_url") or "").strip()
            if url:
                try: st.image(url, use_container_width=True)
                except: st.caption("(image unavailable)")
            else:
                st.caption("(image pending)")
            if m.get("notes"): st.code(m["notes"], language="text")

def render_history():
    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def render_overrides():
    st.subheader("Media Overrides (optional)")
    for sid,s in st.session_state.slots.items():
        with st.expander(f"Override for {s.get('label') or sid}"):
            c1,c2=st.columns(2)
            with c1:
                url=st.text_input("Image URL (http/https)", key=f"url_{sid}")
                if url.startswith("http"):
                    s["media"]["status"]="found"
                    s["media"]["best_image_url"]=url
                    s["media"]["notes"]="override url"
                    st.success("Using URL override.")
            with c2:
                up=st.file_uploader("Upload image", type=["png","jpg","jpeg","webp"], key=f"up_{sid}")
                if up:
                    try:
                        img=Image.open(up); path=os.path.join(MEDIA_DIR, f"{sid}_upload.png"); img.save(path,"PNG")
                        s["media"]["status"]="uploaded"; s["media"]["best_image_url"]=path; s["media"]["notes"]="uploaded"
                        st.success("Using uploaded image.")
                    except:
                        st.error("Upload failed.")

def render_final():
    lines=st.session_state.final_lines
    if not lines: return
    st.subheader("Your Expert Card")
    for sid in st.session_state.slot_order:
        s=st.session_state.slots.get(sid)
        if not s: continue
        label=s.get("label","").strip()
        if not label: continue
        line=lines.get(label,"").strip()
        if not line: continue
        url=(s.get("media",{}).get("best_image_url") or "").strip()
        c1,c2=st.columns([1,2])
        with c1:
            if url:
                try: st.image(url, use_container_width=True)
                except: st.caption("(image unavailable)")
            else: st.caption("(image pending)")
        with c2:
            st.markdown(f"**{label}**")
            st.write(line)

# ========= Main =========
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
st.caption("GPT-5 only • Hosted Web Search • Async media • One short question per turn")

client = get_client()
init_state()

# Preflight
if st.session_state.web_ok is None:
    try:
        _= client.responses.create(
            model=OPENAI_MODEL,
            tools=[{"type":"web_search_preview"}],
            input=[{"role":"user","content":[{"type":"input_text","text":"Return JSON {\"ok\":true}"}]}]
        )
        st.session_state.web_ok=True
    except Exception as e:
        st.session_state.web_ok=False
        st.session_state.web_err=str(e)
if st.session_state.web_ok is False:
    st.error(f"Hosted web_search not available: {st.session_state.web_err}")

orch = Orchestrator()

# Seed: dynamische Eingangsfrage von Agent 1
if not any(m["role"]=="assistant" for m in st.session_state.history):
    opener=agent1_opening_question()
    st.session_state.history.append({"role":"assistant","content": opener})

# Poll media jobs
updated = orch.poll_media()
if updated:
    for sid in updated:
        st.toast(f"Media updated: {sid}", icon="🖼️")

# Render
render_timeline()
render_history()

# Input
user_text = st.chat_input("Your turn…")
if user_text:
    st.session_state.history.append({"role":"user","content": user_text})
    process_user_turn(user_text, orch)
    st.rerun()

# Actions
c1,c2,c3=st.columns(3)
with c1:
    if st.button("✨ Finalize (needs 4 slots)"):
        res=orch.finalize()
        if not res.get("ok"):
            st.warning(res.get("error","Could not finalize"))
        else:
            st.success("Finalized!")
            st.rerun()
with c2:
    if st.button("🔄 Restart"):
        try: st.session_state.executor.shutdown(cancel_futures=True)
        except: pass
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()
with c3:
    if st.button("🧹 Clear Final"):
        st.session_state.final_lines={}
        st.rerun()

render_final()
render_overrides()
