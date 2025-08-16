# app_expert_card_gpt5_3agents_clean.py
# 3 Agents: Chat (Agent 1) · Detect+Verify (Agent 2) · Finalize (Agent 3)
# Agent 2: Items extrahieren (günstig) + Bild über Google CSE holen + Vision-Validierung

import os, json, time, uuid, random, traceback
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from openai import APIStatusError
import requests

APP_TITLE = "🟡 Expert Card — GPT-5 (3 Agents · Google Image + Vision Verify)"
MODEL = os.getenv("OPENAI_GPT5_SNAPSHOT", "gpt-5-2025-08-07")
AGENT2_MODEL = os.getenv("OPENAI_AGENT2_MODEL", "gpt-4o-mini")          # günstig (Text)
AGENT2_VISION_MODEL = os.getenv("OPENAI_AGENT2_VISION_MODEL", "gpt-4o-mini")  # Vision
MEDIA_DIR = os.path.join(os.getcwd(), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

# ---- Controls
OPENAI_MAX_RETRIES = 3
OPENAI_BACKOFF_BASE = 1.8
MAX_CANDIDATES_TO_VERIFY = int(os.getenv("MAX_CANDIDATES_TO_VERIFY", "1"))  # 1 = kein endloses Probieren

# ---- Google CSE env
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX", "").strip()

# ---------- OpenAI client
def client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    return OpenAI(api_key=key)

# ---------- Simple retry wrapper
def call_with_retry(fn: Callable, *args, **kwargs):
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except APIStatusError as e:
            status = getattr(e, "status_code", None)
            if status in (429, 500, 502, 503, 504):
                delay = OPENAI_BACKOFF_BASE ** attempt + random.uniform(0, 0.3)
                time.sleep(delay)
                continue
            raise
        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                delay = OPENAI_BACKOFF_BASE ** attempt + random.uniform(0, 0.3)
                time.sleep(delay)
                continue
            raise
    return fn(*args, **kwargs)

# ---------- Session state
def init_state():
    st.session_state.setdefault("history", [])                # Agent1 ↔ User
    st.session_state.setdefault("slots", {})                  # S1..S4
    st.session_state.setdefault("order", ["S1","S2","S3","S4"])
    st.session_state.setdefault("jobs", {})                   # id -> (sid, Future)
    if "executor" not in st.session_state or st.session_state.get("executor") is None:
        st.session_state.executor = ThreadPoolExecutor(max_workers=4)
    st.session_state.setdefault("seen_entities", [])          # e.g., "book|the little prince"
    st.session_state.setdefault("final_text", "")
    st.session_state.setdefault("used_openers", set())
    st.session_state.setdefault("auto_finalized", False)

# ---------- Utils
def parse_json_loose(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1 and b > a:
        frag = text[a:b+1]
        try:
            return json.loads(frag)
        except Exception:
            return None
    return None

def placeholder_image(text: str, name: str) -> str:
    img = Image.new("RGB", (640, 640), (24, 31, 55))
    d = ImageDraw.Draw(img)
    for r, c in [(260, (57, 96, 199)), (200, (73, 199, 142)), (140, (255, 205, 86))]:
        d.ellipse([(320 - r, 320 - r), (320 + r, 320 + r)], outline=c, width=8)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
    except Exception:
        font = ImageFont.load_default()
    label = (text or "Idea")[:22]
    try:
        bbox = d.textbbox((0, 0), label, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        w, h = d.textsize(label, font=font)
    d.text(((640 - w) // 2, (640 - h) // 2), label, fill=(240, 240, 240), font=font)
    path = os.path.join(MEDIA_DIR, f"{name}_ph.png")
    img.save(path, "PNG")
    return path

def next_free_slot() -> Optional[str]:
    for sid in st.session_state.order:
        if sid not in st.session_state.slots:
            return sid
    return None

# ---------- Agent 1 (Interview)
AGENT1_SYSTEM = """You are Agent 1 — a warm, incisive interviewer.

## Scope
You run the conversation and never call tools. Agent 2 observes items; Agent 3 finalizes.

## Mission
Build a 4-item Expert Card: PUBLIC anchors (book/podcast/person/tool/film) + the user’s personal angle (decisions, habits, examples, trade-offs). Favor variety.

## Style
- Natural, concise (1–3 short sentences). Mirror user language. One primary move per turn: micro-clarify → deepen → pivot → close-and-move.

## Deepen vs Pivot
- Deepen up to ~2 turns if concrete and high-signal; else pivot for diversity.

## Handling inputs
- PUBLIC item present → proceed.
- Private/unusual → ask if public; else request a PUBLIC stand-in.
- Multiple items → pick one now; revisit others later.
- Sensitive → acknowledge briefly; steer back to practice + PUBLIC anchors.

## Stop Condition
When 4 strong slots are covered, say: “Got it — I’ll assemble your 4-point card now.”
"""

def agent1_next_question(history: List[Dict[str, str]]) -> str:
    msgs = [{"role": "system", "content": AGENT1_SYSTEM}]
    if st.session_state.used_openers:
        avoid = "Avoid these phrasings this session: " + "; ".join(list(st.session_state.used_openers))[:600]
        msgs.append({"role": "system", "content": avoid})
    short = history[-6:] if len(history) > 6 else history
    msgs += short
    resp = call_with_retry(client().chat.completions.create, model=MODEL, messages=msgs, temperature=0)
    q = (resp.choices[0].message.content or "").strip()
    if "\n" in q:
        q = q.split("\n")[0].strip()
    if not q.endswith("?"):
        q = q.rstrip(".! ") + "?"
    st.session_state.used_openers.add(q.lower()[:72])
    return q

# ---------- Agent 2 (Detector — text only)
AGENT2_SYSTEM = """You are Agent 2 (detector).
From assistant_question + user_reply extract ZERO OR MORE PUBLIC items with types: book | podcast | person | tool | film.
Return strict JSON:
{"detected": true|false, "items":[{"entity_type":"book|podcast|person|tool|film","entity_name":"..."}]}
Rules: Use the question only for disambiguation. If none present → {"detected": false}. No extra text.
"""

def agent2_detect_items(last_q: str, user_reply: str) -> Dict[str, Any]:
    payload = {"assistant_question": last_q or "", "user_reply": user_reply or ""}
    try:
        resp = call_with_retry(
            client().chat.completions.create,
            model=AGENT2_MODEL,
            messages=[
                {"role": "system", "content": AGENT2_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            temperature=0
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = parse_json_loose(raw)
        if isinstance(data, dict):
            return data
        return {"detected": False, "reason": "no-parse"}
    except Exception as e:
        return {"detected": False, "reason": f"error: {e}"}

# ---------- Google CSE (image)
BLOCKED_HOSTS = {
    "pinterest.", "alamy.", "shutterstock.", "istockphoto.", "gettyimages.",
    "dreamstime.", "depositphotos.", "123rf.", "vectorstock.", "freepik.", "adobe."
}

def _host(url: str) -> str:
    try:
        return requests.utils.urlparse(url).netloc.lower()
    except Exception:
        return ""

def google_image_candidates(query: str, *, site: Optional[str]=None, img_type: Optional[str]=None, num: int=5) -> List[Dict[str, Any]]:
    """Return up to `num` candidate dicts with direct image links."""
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_CX:
        return []
    params = {
        "key": GOOGLE_CSE_API_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "searchType": "image",
        "num": max(1, min(num, 10)),
        "safe": "active",
        "imgSize": "large"
    }
    if site:
        params["siteSearch"] = site
    if img_type:
        params["imgType"] = img_type
    url = "https://www.googleapis.com/customsearch/v1?" + urlencode(params)
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        items = data.get("items") or []
        out = []
        for it in items:
            link = it.get("link") or ""
            imeta = it.get("image") or {}
            w, h = int(imeta.get("width", 0) or 0), int(imeta.get("height", 0) or 0)
            mime = it.get("mime", "")
            host = _host(link)
            if any(b in host for b in BLOCKED_HOSTS):
                continue
            if not link.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and not (mime and ("image/" in mime)):
                continue
            if max(w, h) < 300:
                continue
            out.append({
                "link": link,
                "contextLink": imeta.get("contextLink") or it.get("image", {}).get("contextLink") or it.get("link"),
                "mime": mime,
                "width": w,
                "height": h,
                "byteSize": imeta.get("byteSize"),
                "host": host
            })
        return out
    except Exception:
        return []

def query_hint(etype: str, name: str) -> Dict[str, Optional[str]]:
    et = (etype or "").lower()
    if et == "person":
        return {"q": f'{name} official portrait', "img_type": "face", "site": None}
    if et == "book":
        return {"q": f'{name} book cover', "img_type": None, "site": None}
    if et == "podcast":
        return {"q": f'{name} podcast cover', "img_type": None, "site": None}
    if et == "tool":
        return {"q": f'{name} logo', "img_type": None, "site": None}
    if et == "film":
        return {"q": f'{name} poster', "img_type": None, "site": None}
    return {"q": name, "img_type": None, "site": None}

# ---------- Vision verification
VISION_SYSTEM = """You are an image verifier. Output strict JSON only.
Decide if the image matches the referenced PUBLIC item and type for use on an expert card.
Accept only if it is plausibly the correct item (not stock/watermarked/meme/wrong subject).
JSON schema: {"ok":true|false,"confidence":0..1,"reason":"short"}"""

def verify_image_with_vision(etype: str, name: str, link: str, context: str) -> Dict[str, Any]:
    # Minimal, deterministic check
    try:
        prompt = (
            f"Type: {etype}\n"
            f"Name: {name}\n"
            f"Context (why relevant): {context[:240]}\n"
            "Does this image plausibly match the item and type (e.g., book cover, podcast cover, person portrait, tool logo, film poster)?"
            " Only say true if it's clearly correct or close enough for an expert-card thumbnail."
        )
        resp = call_with_retry(
            client().chat.completions.create,
            model=AGENT2_VISION_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": VISION_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": link}}
                ]}
            ]
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = parse_json_loose(raw)
        if isinstance(data, dict) and "ok" in data:
            # Clamp confidence
            try:
                c = float(data.get("confidence", 0))
                data["confidence"] = max(0.0, min(1.0, c))
            except Exception:
                data["confidence"] = 0.0
            return data
        return {"ok": False, "confidence": 0.0, "reason": "no-parse"}
    except Exception as e:
        return {"ok": False, "confidence": 0.0, "reason": f"error: {e}"}

# ---------- Agent 3 (Finalizer)
FINALIZER_SYSTEM = """You are Agent 3 — Finalizer.
Create an Expert Card with exactly 4 items from:
- The conversation (assistant questions + user answers).
- The current slots (labels + image URLs if any).
Rules:
- Each line: '- Label: 1–2 short sentences' grounded in THIS user.
- Prefer PUBLIC items; if fewer than 4, fill with the user's stated practices/principles.
- No fluff, no generic facts, no references to these instructions.
"""

def agent3_finalize(history: List[Dict[str, str]], slots: Dict[str, Dict[str, Any]]) -> str:
    convo = []
    for m in history[-24:]:
        if m["role"] == "assistant":
            convo.append("Q: " + m["content"])
        else:
            convo.append("A: " + m["content"])
    convo_text = "\n".join(convo)

    slot_lines = []
    for sid in ["S1", "S2", "S3", "S4"]:
        s = slots.get(sid)
        if not s:
            continue
        lab = s.get("label", "").strip()
        img = (s.get("media", {}).get("best_image_url") or "").strip()
        if lab:
            slot_lines.append(f"{sid}: {lab} | image={img or 'n/a'}")
    slots_text = "\n".join(slot_lines) if slot_lines else "none"

    msgs = [
        {"role": "system", "content": FINALIZER_SYSTEM},
        {"role": "user", "content": f"Transcript:\n{convo_text}\n\nSlots:\n{slots_text}"}
    ]
    resp = call_with_retry(client().chat.completions.create, model=MODEL, messages=msgs, temperature=0)
    return (resp.choices[0].message.content or "").strip()

# ---------- Orchestrator
class Orchestrator:
    def __init__(self):
        if "seen_entities" not in st.session_state:
            st.session_state.seen_entities = []
        self.slots = st.session_state.slots
        self.jobs = st.session_state.jobs
        self.exec = st.session_state.executor
        self.seen = st.session_state.seen_entities

    def upsert(self, sid: str, label: str, media: Dict[str, Any]):
        s = self.slots.get(sid, {"slot_id": sid, "label": "", "media": {"status": "pending", "best_image_url": "", "candidates": [], "notes": ""}})
        s["label"] = label[:160] if label else sid
        m = s.get("media", {})
        m.setdefault("status", "pending")
        m.setdefault("best_image_url", "")
        m.setdefault("candidates", [])
        m.setdefault("notes", "")
        m.update(media or {})
        s["media"] = m
        self.slots[sid] = s

    def schedule_watch(self, last_q: str, reply: str):
        jid = str(uuid.uuid4())[:8]
        fut = self.exec.submit(self._job, last_q, reply, list(self.seen))
        self.jobs[jid] = ("TBD", fut)
        st.toast("🛰️ Agent 2 is extracting + vision-verifying…", icon="🛰️")

    def _job(self, last_q: str, reply: str, seen_snapshot: List[str]) -> Dict[str, Any]:
        try:
            det = agent2_detect_items(last_q, reply)
            if not det.get("detected"):
                return {"status": "skip"}
            items = det.get("items") or []
            results: List[Dict[str, Any]] = []

            # Short context for the verifier (blend last Q & A)
            context = (last_q or "")[:160] + " || " + (reply or "")[:200]

            for item in items:
                etype = (item.get("entity_type") or "").lower().strip()
                ename = (item.get("entity_name") or "").strip()
                if not etype or not ename:
                    continue

                key = f"{etype}|{ename.lower()}"
                if key in self.seen:
                    continue

                # Google image candidates (direct URLs)
                hint = query_hint(etype, ename)
                candidates = google_image_candidates(hint["q"], site=hint["site"], img_type=hint["img_type"], num=max(1, MAX_CANDIDATES_TO_VERIFY))

                if not candidates:
                    continue

                # Try limited number of candidates with vision verification
                chosen = None
                tried = 0
                for cand in candidates:
                    if tried >= MAX_CANDIDATES_TO_VERIFY:
                        break
                    tried += 1
                    v = verify_image_with_vision(etype, ename, cand["link"], context)
                    if v.get("ok") and v.get("confidence", 0) >= 0.5:
                        chosen = (cand, v)
                        break

                if not chosen:
                    # no suitable image → skip slot (keine falschen Treffer)
                    continue

                # Create slot
                self.seen.append(key)
                cand, v = chosen
                label_hint = {
                    "book": "Must-Read",
                    "podcast": "Podcast",
                    "person": "Role Model",
                    "tool": "Go-to Tool",
                    "film": "Influence",
                }.get(etype, "Item")
                label = f"{label_hint} — {ename}"

                media = {
                    "status": "found",
                    "best_image_url": cand["link"],
                    "candidates": [{
                        "url": cand["link"],
                        "page_url": cand.get("contextLink") or "",
                        "source": cand.get("host") or "",
                        "confidence": float(v.get("confidence", 0)),
                        "reason": f"Vision OK: {v.get('reason','')[:80]}"
                    }],
                    "notes": f"{cand.get('host','')} · {cand.get('width','?')}×{cand.get('height','?')}"
                }

                sid = next_free_slot()
                if not sid:
                    results.append({"status": "full", "label": label})
                    break
                results.append({"status": "ok", "sid": sid, "label": label, "media": media})

            return {"status": "batch", "items": results}
        except Exception as e:
            return {"status": "error", "error": str(e), "trace": traceback.format_exc()[:1200]}

    def poll(self) -> List[str]:
        updated, rm = [], []
        for jid, (sid, fut) in list(self.jobs.items()):
            if fut.done():
                rm.append(jid)
                try:
                    res = fut.result()
                except Exception:
                    continue
                if res.get("status") == "batch":
                    for it in res.get("items", []):
                        if it.get("status") == "ok":
                            self.upsert(it["sid"], it["label"], it["media"])
                            updated.append(it["sid"])
        for jid in rm:
            del self.jobs[jid]
        return updated

# ---------- Render
def render_slots():
    slots = st.session_state.slots
    filled = len([s for s in slots.values() if (s.get("label") or "").strip()])
    st.progress(min(1.0, filled / 4), text=f"Progress: {filled}/4")
    cols = st.columns(4)
    for i, sid in enumerate(st.session_state.order):
        s = slots.get(sid)
        with cols[i]:
            st.markdown(f"**{(s or {}).get('label') or sid}**")
            if not s:
                st.caption("(empty)")
                continue
            m = s.get("media", {})
            st.caption(f"status: {m.get('status', 'pending')}")
            best = (m.get("best_image_url") or "").strip()
            if best:
                try:
                    st.image(best, use_container_width=True)
                except Exception:
                    st.caption("(image unavailable)")
            else:
                st.caption("(image pending)")
            notes = (m.get("notes") or "").strip()
            if notes:
                st.code(notes, language="text")

def render_history():
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
                if url.startswith("http"):
                    s["media"]["status"] = "found"
                    s["media"]["best_image_url"] = url
                    s["media"]["notes"] = "override url"
                    st.success("Using URL override.")
            with c2:
                up = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"], key=f"up_{sid}")
                if up:
                    try:
                        path = os.path.join(MEDIA_DIR, f"{sid}_upload.png")
                        Image.open(up).save(path, "PNG")
                        s["media"]["status"] = "uploaded"
                        s["media"]["best_image_url"] = path
                        s["media"]["notes"] = "uploaded file"
                        st.success("Using uploaded image.")
                    except Exception:
                        st.error("Could not read image.")

# ---------- Main
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
st.caption("Agent 2 extrahiert Items, holt Bild via Google CSE und **validiert per Vision**. Slots zählen nur verifizierte Bilder.")

init_state()

orch = Orchestrator()

# Poll Agent 2 async jobs
for sid in orch.poll():
    st.toast(f"🖼️ Media updated: {sid}", icon="🖼️")

# First opener from Agent 1
if not st.session_state.history:
    opener = agent1_next_question([])
    st.session_state.history.append({"role": "assistant", "content": opener})
    st.rerun()

# Render UI
render_slots()
render_history()

# Auto-finalize when 4 slots filled
if not st.session_state.auto_finalized:
    if all(sid in st.session_state.slots and (st.session_state.slots[sid].get("label") or "").strip() for sid in st.session_state.order):
        st.session_state.final_text = agent3_finalize(st.session_state.history, st.session_state.slots)
        st.session_state.auto_finalized = True
        st.toast("✅ Auto-finalized your Expert Card.", icon="✨")

# Input handling
user_text = st.chat_input("Your turn…")
if user_text:
    st.session_state.history.append({"role": "user", "content": user_text})

    # Find last Agent 1 question
    last_q = ""
    for m in reversed(st.session_state.history[:-1]):
        if m["role"] == "assistant":
            last_q = m["content"]
            break

    # Agent 2: detect → google image → vision verify → create slot
    orch.schedule_watch(last_q, user_text)

    # Agent 1: next question
    nxt = agent1_next_question(st.session_state.history)
    st.session_state.history.append({"role": "assistant", "content": nxt})

    # Optional handoff phrase trigger
    if "assemble your 4-point card now" in nxt.lower() or "assemble your 4-point" in nxt.lower():
        st.session_state.final_text = agent3_finalize(st.session_state.history, st.session_state.slots)
        st.session_state.auto_finalized = True

    st.rerun()

# Actions
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("✨ Finalize"):
        st.session_state.final_text = agent3_finalize(st.session_state.history, st.session_state.slots)
        st.session_state.auto_finalized = True
        st.success("Finalized.")
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
        st.session_state.final_text = ""
        st.session_state.auto_finalized = False
        st.rerun()

# Final card
if st.session_state.final_text:
    st.subheader("Your Expert Card")
    st.write(st.session_state.final_text)

render_overrides()
