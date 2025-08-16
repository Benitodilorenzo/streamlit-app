# app_expert_card_gpt5_3agents_clean.py
# 3 Agents: Chat (Agent 1) · Search (Agent 2) · Finalize (Agent 3)
# Uses Google Custom Search (image) for real images; Agent 2 runs cheap LLM extraction + HTTP image search.

import os, json, time, uuid, random, traceback, re
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import requests
from urllib.parse import urlparse

from openai import OpenAI
from openai import APIStatusError

APP_TITLE = "🟡 Expert Card — GPT-5 (3 Agents · Google Image Search · Async)"
MODEL = os.getenv("OPENAI_GPT5_SNAPSHOT", "gpt-5-2025-08-07")
AGENT2_MODEL = os.getenv("OPENAI_AGENT2_MODEL", "gpt-4o-mini")  # günstig für Extraction
MEDIA_DIR = os.path.join(os.getcwd(), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

# ---- Google Image Search (Programmable Search)
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ---- Cost/robustness
MAX_SEARCHES_PER_RUN = 1         # Agent 2: max Bildsuchen pro Turn
OPENAI_MAX_RETRIES = 3           # Retry count for rate limits / 5xx
OPENAI_BACKOFF_BASE = 1.8        # Exponential backoff base

# ---------- OpenAI client
def client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    return OpenAI(api_key=key)

# ---------- Simple retry wrapper for OpenAI calls
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

# ---------- Session state (idempotent)
def init_state():
    st.session_state.setdefault("init", True)
    st.session_state.setdefault("history", [])                # Agent1 ↔ User
    st.session_state.setdefault("slots", {})                  # S1..S4
    st.session_state.setdefault("order", ["S1","S2","S3","S4"])
    st.session_state.setdefault("jobs", {})                   # id -> (sid, Future)
    if "executor" not in st.session_state or st.session_state.get("executor") is None:
        st.session_state.executor = ThreadPoolExecutor(max_workers=4)
    st.session_state.setdefault("seen_entities", [])          # e.g., "book|antifragile"
    st.session_state.setdefault("final_text", "")
    st.session_state.setdefault("img_search_ok", None)
    st.session_state.setdefault("img_search_err", "")
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
    # Wird nur bei Overrides genutzt – Agent 2 legt keinen Slot ohne echtes Bild an.
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

def hostname(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""

# ---------- Google Image Search (JSON API)
def google_image_search(query: str, kind: str) -> Optional[Dict[str, str]]:
    """Return {'url', 'page_url', 'source'} or None."""
    if not GOOGLE_CSE_ID or not GOOGLE_API_KEY:
        return None

    q = query.strip()
    # leichte Query-Hints je nach Typ
    if kind == "book":
        # häufig sinnvoll, um Cover statt Fan-Art zu bekommen
        if "cover" not in q.lower():
            q = f"{q} book cover"
    elif kind == "person":
        # Portraits eher als Logos
        if "portrait" not in q.lower():
            q = f"{q} portrait"
    elif kind == "podcast":
        if "podcast" not in q.lower():
            q = f"{q} podcast"
    elif kind == "tool":
        # nichts erzwingen – oft Produktshots
        pass
    elif kind == "film":
        if "film" not in q.lower() and "movie" not in q.lower():
            q = f"{q} film"

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": q,
        "searchType": "image",
        "num": 5,                 # bis zu 5 Treﬀer prüfen
        "safe": "active",
        "cx": GOOGLE_CSE_ID,
        "key": GOOGLE_API_KEY,
        # Optional: Qualitätsfilter
        "imgType": "photo",
        "imgSize": "large",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        items = data.get("items") or []
        exts = (".jpg", ".jpeg", ".png", ".webp")
        for it in items:
            link = (it.get("link") or "").strip()
            mime = (it.get("mime") or "").lower()
            if (link.lower().endswith(exts)) or mime.startswith("image/"):
                page = (it.get("image", {}) or {}).get("contextLink") or it.get("displayLink") or link
                return {
                    "url": link,
                    "page_url": page,
                    "source": hostname(page) or hostname(link)
                }
        # Falls nichts mit sauberem Bild-Link: nimm den ersten „link“, wenn vorhanden
        if items:
            it = items[0]
            link = (it.get("link") or "").strip()
            if link:
                page = (it.get("image", {}) or {}).get("contextLink") or it.get("displayLink") or link
                return {"url": link, "page_url": page, "source": hostname(page) or hostname(link)}
    except Exception:
        return None
    return None

# ---------- Preflight (prüft nur, ob Google Keys gesetzt sind)
def preflight():
    if st.session_state.img_search_ok is not None:
        return
    if not GOOGLE_CSE_ID or not GOOGLE_API_KEY:
        st.session_state.img_search_ok = False
        st.session_state.img_search_err = "GOOGLE_CSE_ID or GOOGLE_API_KEY missing"
    else:
        st.session_state.img_search_ok = True

# ---------- Agent 1 (Interview)
AGENT1_SYSTEM = """You are Agent 1 — a warm, incisive interviewer.

## Identity & Scope
You are Agent 1. You run the conversation and never call tools.
- Agent 2 observes each turn and handles web images for PUBLIC items (book/podcast/person/tool/film).
- Agent 3 will finalize the 4-point card when you signal the handoff.

## Mission
Build a 4-item Expert Card combining:
1) PUBLIC anchors (book/podcast/person/tool/film), and
2) The user’s personal angle (decisions, habits, examples, trade-offs).
Favor variety across types.

## Output Style (visible)
- Natural, concise (1–3 short sentences). Mirror user language. Warm, specific, no fluff.
- One primary move per turn: micro-clarify → deepen → pivot → close-and-move.
- If the user asks you something, answer in ≤1 sentence, then continue.

## Deepen vs Pivot
- Deepen up to ~2 turns if concrete and high-signal; aim for specific decisions, habits, before/after, in-the-wild examples, accepted trade-offs.
- Pivot if vague/repetitive, you already asked ~2 on this item, or you want diversity.

## Handling inputs
- PUBLIC item present → proceed; Agent 2 will search on its own. Don’t re-confirm what the user already gave unless ambiguity blocks a meaningful question.
- Unusual/private/non-technical item → briefly acknowledge positively and clarify if it’s a public reference (e.g., “Is this someone others could look up, or more of a personal figure?”).
  - If public → one micro-clarifier (name/title/host) if needed, then continue as usual.
  - If private → ask for a PUBLIC stand-in (book/person/tool/podcast/film) that best represents that influence, then pivot.
- Meta/process-only → ask for one PUBLIC stand-in (book/person/tool/podcast/film).
- Multiple items in one reply → pick one now; you may revisit others later.
- Sensitive/emotional → acknowledge briefly; steer back to practice and PUBLIC anchors.

## Diversity Goal
Prefer a mix (e.g., books + a podcast + a person/tool/film). Nudge gently.

## Opening & Flow
Vary openings; reasonable examples include:
- “What’s one book/podcast/person/tool/film that genuinely shifted how you work — and in what way?”
- “Think of the last 12 months: which public influence most changed your decisions?”
- “If someone wanted to think like you in Data/AI, what one public thing should they start with?”
Avoid repeating your own earlier phrasing within this session.

## Stop Condition (Handoff)
When you judge 4 strong slots are covered, say a short line like:
“Got it — I’ll assemble your 4-point card now.”
The system will route to Agent 3.

## Guardrails
Don’t reveal internal agents/tools. Keep it professional. Mirror language switching.

## What you do each turn
1) Read latest user message + light memory of prior slots.
2) Decide one move (micro-clarify, deepen, pivot, close-and-move).
3) Ask one focused question (optionally with a brief synthesis).
4) Continue until 4 good items → give the handoff line.
"""

def agent1_next_question(history: List[Dict[str, str]]) -> str:
    msgs = [{"role": "system", "content": AGENT1_SYSTEM}]
    if st.session_state.used_openers:
        avoid = "Avoid these phrasings this session: " + "; ".join(list(st.session_state.used_openers))[:600]
        msgs.append({"role": "system", "content": avoid})
    short = history[-6:] if len(history) > 6 else history
    msgs += short
    resp = call_with_retry(client().chat.completions.create, model=MODEL, messages=msgs)
    q = (resp.choices[0].message.content or "").strip()
    if "\n" in q:
        q = q.split("\n")[0].strip()
    if not q.endswith("?"):
        q = q.rstrip(".! ") + "?"
    st.session_state.used_openers.add(q.lower()[:72])
    return q

# ---------- Agent 2 (Extractor; Google image search)
AGENT2_SYSTEM = """You are Agent 2.

Task:
- From assistant_question + user_reply extract 0..N PUBLIC items (book|podcast|person|tool|film).
- Return canonical names (e.g., map 'antifragility by taleb' -> 'Antifragile' by Nassim Taleb).
- Deduplicate via keys 'type|normalized_name' (lowercase).
- Do NOT do web searches. Only return clean JSON; the app will fetch images.

Output ONLY strict JSON:
{
  "detected": true|false,
  "items": [
    {
      "entity_type": "book|podcast|person|tool|film",
      "entity_name": "...",     // canonical title/name
      "display_name": "...",    // optional: prettier label (title + author/host)
      "reason": "one short sentence"
    }
  ]
}

Rules:
- If nothing present → {"detected": false}
- No prose outside JSON.
"""

def agent2_extract_items(last_q: str, user_reply: str, seen_entities: List[str]) -> Dict[str, Any]:
    payload = {
        "assistant_question": last_q or "",
        "user_reply": user_reply or "",
        "seen_entities": list(seen_entities or [])
    }
    msgs = [
        {"role": "system", "content": AGENT2_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]
    try:
        resp = call_with_retry(
            client().chat.completions.create,
            model=AGENT2_MODEL,
            messages=msgs,
            temperature=0.1
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = parse_json_loose(raw)
        if isinstance(data, dict):
            return data
        return {"detected": False, "reason": "no-parse"}
    except Exception as e:
        return {"detected": False, "reason": f"error: {e}", "trace": traceback.format_exc()[:600]}

# ---------- Agent 3 (Finalizer)
FINALIZER_SYSTEM = """You are Agent 3 — Finalizer.
Create an Expert Card with exactly 4 items from:
- The conversation (assistant questions + user answers).
- The current slots (labels + image URLs if any).
Rules:
- Each line: '- Label: 1–2 short sentences' grounded in THIS user.
- Prefer PUBLIC items; if fewer than 4, fill with the user's stated practices/principles.
- No fluff, no generic Wikipedia facts, no references to these instructions.
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
    resp = call_with_retry(client().chat.completions.create, model=MODEL, messages=msgs)
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
        s = self.slots.get(sid, {
            "slot_id": sid,
            "label": "",
            "media": {"status": "pending", "best_image_url": "", "candidates": [], "notes": ""}
        })
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
        st.toast("🛰️ Agent 2 is scanning for public items…", icon="🛰️")

    def _job(self, last_q: str, reply: str, seen_snapshot: List[str]) -> Dict[str, Any]:
        try:
            data = agent2_extract_items(last_q, reply, seen_snapshot)
            if not data.get("detected"):
                return {"status": "skip"}

            items = data.get("items") or []
            results: List[Dict[str, Any]] = []
            searches_done = 0

            for item in items:
                etype = (item.get("entity_type") or "").lower().strip()
                ename = (item.get("entity_name") or "").strip()
                dname = (item.get("display_name") or ename).strip()

                if not etype or not ename:
                    continue

                key = f"{etype}|{ename.lower()}"
                if key in self.seen:
                    continue

                if searches_done >= MAX_SEARCHES_PER_RUN:
                    # Wir legen KEINEN Slot an, wenn kein echtes Bild – und wir haben das Kontingent ausgeschöpft
                    continue

                # echte Bildsuche via Google
                hit = google_image_search(ename, etype)
                if not hit or not (hit.get("url") or "").strip():
                    # Kein Bild → kein Slot
                    continue

                searches_done += 1
                # Erst jetzt als "gesehen" markieren (damit nur echte Treffer zählen)
                self.seen.append(key)

                label_hint = {
                    "book": "Must-Read",
                    "podcast": "Podcast",
                    "person": "Role Model",
                    "tool": "Go-to Tool",
                    "film": "Influence",
                }.get(etype, "Item")
                label = f"{label_hint} — {dname or ename}"

                url = hit.get("url", "").strip()
                page = hit.get("page_url", "").strip()
                src = hit.get("source", "").strip()

                media = {
                    "status": "found",
                    "best_image_url": url,
                    "candidates": [{"url": url, "page_url": page, "source": src}],
                    "notes": f"google image | {src}"
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
        for jid, (_, fut) in list(self.jobs.items()):
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
st.caption("Günstige LLM-Orchestrierung • Google Image Search (Programmable Search) • Agent 2 erzeugt Slots nur bei echten Bildtreffern")

init_state()
preflight()
if st.session_state.img_search_ok is False:
    st.error(f"Image search not configured: {st.session_state.img_search_err}")

# FIRST OPENER sofort setzen
if not st.session_state.history:
    opener = agent1_next_question([])
    st.session_state.history.append({"role": "assistant", "content": opener})
    st.rerun()

orch = Orchestrator()

# Poll Agent 2 async jobs
for sid in orch.poll():
    st.toast(f"🖼️ Media updated: {sid}", icon="🖼️")

# Render UI
render_slots()
render_history()

# Auto-finalize: wenn 4 Slots gefüllt
if not st.session_state.auto_finalized:
    if all(sid in st.session_state.slots and (st.session_state.slots[sid].get("label") or "").strip() for sid in st.session_state.order):
        st.session_state.final_text = agent3_finalize(st.session_state.history, st.session_state.slots)
        st.session_state.auto_finalized = True
        st.toast("✅ Auto-finalized your Expert Card.", icon="✨")

# Input handling
user_text = st.chat_input("Your turn…")
if user_text:
    st.session_state.history.append({"role": "user", "content": user_text})

    # Letzte Frage des Assistenten (Agent 1) finden
    last_q = ""
    for m in reversed(st.session_state.history[:-1]):
        if m["role"] == "assistant":
            last_q = m["content"]
            break

    # Agent 2: Extrahieren + Google-Bildsuche (nur letzte Q↔A), dedupe via seen_entities
    orch.schedule_watch(last_q, user_text)

    # Agent 1: nächste Frage
    nxt = agent1_next_question(st.session_state.history)
    st.session_state.history.append({"role": "assistant", "content": nxt})

    # Handoff-Trigger (falls Agent 1 es ausspricht)
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
