# app_expert_card_gpt5_3agents_clean.py
# 3 Agents: Chat (Agent 1) · Search (Agent 2) · Finalize (Agent 3)
# Debug sichtbar im Frontend: Agent-2 Steps + Google Image Search
# Nur Debug-Patches; Agent 1 & Agent 2 Prompts wie besprochen.

import os, json, time, uuid, random, traceback, requests
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from openai import APIStatusError
from requests.exceptions import HTTPError

APP_TITLE = "🟡 Expert Card — GPT-5 (3 Agents · Google Image API · Async)"
MODEL = os.getenv("OPENAI_GPT5_SNAPSHOT", "gpt-5-2025-08-07")
AGENT2_MODEL = os.getenv("OPENAI_AGENT2_MODEL", "gpt-4o-mini")  # günstig für Extraktion + Vision
MEDIA_DIR = os.path.join(os.getcwd(), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

# ---- Google Programmable Search (CSE)
GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY", "").strip()
GOOGLE_CSE_CX  = os.getenv("GOOGLE_CSE_CX", "").strip()

# ---- Token/Cost controls
MAX_SEARCHES_PER_RUN = 1
OPENAI_MAX_RETRIES = 3
OPENAI_BACKOFF_BASE = 1.8

# ---------- OpenAI client
def client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    return OpenAI(api_key=key)

# ---------- Retry wrapper
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
    st.session_state.setdefault("init", True)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("slots", {})
    st.session_state.setdefault("order", ["S1","S2","S3","S4"])
    st.session_state.setdefault("jobs", {})
    if "executor" not in st.session_state or st.session_state.get("executor") is None:
        st.session_state.executor = ThreadPoolExecutor(max_workers=4)
    st.session_state.setdefault("seen_entities", [])
    st.session_state.setdefault("final_text", "")
    st.session_state.setdefault("used_openers", set())
    st.session_state.setdefault("auto_finalized", False)
    # Debugging
    st.session_state.setdefault("debug_agent2", [])
    st.session_state.setdefault("debug_limit", 80)

# ---------- Debug helpers (thread-safe via buffer)
def debug_emit(event: Dict[str, Any], buffer: Optional[list] = None):
    """In Worker-Threads: in lokalen Buffer; im Main-Thread: direkt in session_state."""
    e = dict(event)
    e["ts"] = time.strftime("%H:%M:%S")
    if buffer is not None:
        buffer.append(e)
        return
    if "debug_agent2" not in st.session_state:
        st.session_state["debug_agent2"] = []
    st.session_state.debug_agent2.append(e)
    limit = st.session_state.get("debug_limit", 80)
    if len(st.session_state.debug_agent2) > limit:
        st.session_state.debug_agent2 = st.session_state.debug_agent2[-limit:]

def debug_log_agent2(event: Dict[str, Any]):
    debug_emit(event, buffer=None)

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

# ---------- Agent 1 (Interview) — ORIGINAL PROMPT
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
- Remember: this is a profile interview, not a metrics deep-dive. Do not ask about KPIs, business performance metrics, or detailed professional analytics. Keep follow-ups at most 1–2 levels deep.

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
    if st.session_state.get("used_openers"):
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
    st.session_state.setdefault("used_openers", set()).add(q.lower()[:72])
    return q

# ---------- Agent 2 (Extractor ONLY; keine Tools) — ORIGINAL
AGENT2_SYSTEM = """You are Agent 2.

Task:
- From assistant_question + user_reply extract 0..N PUBLIC items (book|podcast|person|tool|film).
- For each item, return {entity_type, entity_name}. Do NOT search the web. Do NOT invent.
- Dedupe by normalized lowercase name per type.

Output ONLY strict JSON:
{
  "detected": true|false,
  "items": [
    {"entity_type":"book|podcast|person|tool|film","entity_name":"..."}
  ]
}

Rules:
- If no item present → {"detected": false}
- Return ONLY JSON.
"""

def agent2_extract_items(last_q: str, user_reply: str, seen_entities: List[str], dbg: Optional[list] = None) -> Dict[str, Any]:
    payload = {
        "assistant_question": last_q or "",
        "user_reply": user_reply or "",
        "seen_entities": list(seen_entities or [])
    }
    try:
        debug_emit({"ev":"extract_start", "q_len": len(last_q), "a_len": len(user_reply)}, dbg)
        resp = call_with_retry(
            client().chat.completions.create,
            model=AGENT2_MODEL,
            messages=[
                {"role":"system","content": AGENT2_SYSTEM},
                {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
            ]
        )
        raw = (resp.choices[0].message.content or "").strip()
        debug_emit({"ev":"extract_raw", "preview": raw[:240]}, dbg)
        data = parse_json_loose(raw)
        if isinstance(data, dict):
            debug_emit({"ev":"extract_parsed", "detected": bool(data.get("detected")), "count": len(data.get("items") or [])}, dbg)
            return data
        debug_emit({"ev":"extract_noparse"}, dbg)
        return {"detected": False, "reason": "no-parse"}
    except Exception as e:
        debug_emit({"ev":"extract_error", "error": str(e)}, dbg)
        return {"detected": False, "reason": f"error: {e}", "trace": traceback.format_exc()[:600]}

# ---------- Google CSE: Image Search (nur Debug erweitert)
def google_image_search(query: str, num: int = 4, dbg: Optional[list] = None) -> List[Dict[str, str]]:
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        debug_emit({"ev":"cse_keys_missing", "query": query, "key_set": bool(GOOGLE_CSE_KEY), "cx_set": bool(GOOGLE_CSE_CX)}, dbg)
        return []
    try:
        params = {
            "q": query,
            "searchType": "image",
            "num": max(1, min(num, 10)),
            "safe": "active",
            "fields": "items(link,contextLink),error",
            "key": GOOGLE_CSE_KEY,
            "cx": GOOGLE_CSE_CX,
        }
        debug_emit({"ev":"cse_start", "query": query, "params": {"num": params["num"], "safe": "active"}}, dbg)
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=12)
        try:
            r.raise_for_status()
        except HTTPError as he:
            # Debug: HTTP-Fehler + Body (JSON oder Text)
            try:
                err_json = r.json()
            except Exception:
                err_json = {"text": r.text[:400]}
            debug_emit({"ev":"cse_http_error", "status": r.status_code, "error": str(he), "body": err_json}, dbg)
            return []
        data = r.json()
        if "error" in data:
            debug_emit({"ev":"cse_api_error", "query": query, "api_error": data.get("error")}, dbg)
            return []
        items = data.get("items", []) or []
        out = []
        for it in items:
            link = (it.get("link") or "").strip()
            ctx  = (it.get("contextLink") or "").strip()
            if link and any(link.lower().endswith(ext) for ext in (".jpg",".jpeg",".png",".webp")):
                out.append({"url": link, "page_url": ctx})
        debug_emit({"ev":"cse_done", "query": query, "returned": len(out)}, dbg)
        return out
    except Exception as e:
        debug_emit({"ev":"cse_error", "query": query, "error": str(e)}, dbg)
        return []

# ---------- Vision: Validate image vs. context
def validate_image_with_context(image_url: str, entity_type: str, entity_name: str, q_text: str, a_text: str, dbg: Optional[list] = None) -> Dict[str, Any]:
    sys = (
        "You are an image verifier. Respond with STRICT JSON like "
        "{\"ok\":true/false,\"reason\":\"...\"}. "
        "Check if the image plausibly depicts the requested public item "
        "(book/podcast/person/tool/film) given the context. "
        "Be strict about wrong people/logos, but flexible with cover art variants."
    )
    user_text = (
        f"Item type: {entity_type}\n"
        f"Item name: {entity_name}\n"
        f"Context Q: {q_text[:300]}\n"
        f"Context A: {a_text[:500]}\n"
        "Does this image plausibly match the item?"
    )
    try:
        resp = call_with_retry(
            client().chat.completions.create,
            model=AGENT2_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ]
        )
        txt = (resp.choices[0].message.content or "").strip()
        data = parse_json_loose(txt)
        ok = bool(data.get("ok")) if isinstance(data, dict) else False
        reason = (data.get("reason") if isinstance(data, dict) else "") or ""
        debug_emit({"ev":"vision_check", "item": f"{entity_type}|{entity_name}", "ok": ok, "reason": reason[:160], "img": image_url[:140]}, dbg)
        if isinstance(data, dict) and "ok" in data:
            return {"ok": ok, "reason": reason[:200]}
    except Exception as e:
        debug_emit({"ev":"vision_error", "item": f"{entity_type}|{entity_name}", "error": str(e)}, dbg)
    return {"ok": False, "reason": "unverifiable"}

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
        debug_emit({"ev":"job_scheduled", "jid": jid})

    def _job(self, last_q: str, reply: str, seen_snapshot: List[str]) -> Dict[str, Any]:
        dbg = []  # lokaler Buffer (Worker)
        try:
            data = agent2_extract_items(last_q, reply, seen_snapshot, dbg=dbg)
            if not data.get("detected"):
                debug_emit({"ev":"job_skip_no_items"}, dbg)
                return {"status": "skip", "debug": dbg}

            items = data.get("items") or []
            debug_emit({"ev":"job_items", "count": len(items)}, dbg)
            processed = 0
            results: List[Dict[str, Any]] = []

            for item in items:
                if processed >= MAX_SEARCHES_PER_RUN:
                    debug_emit({"ev":"job_cap_reached", "max": MAX_SEARCHES_PER_RUN}, dbg)
                    break
                etype = (item.get("entity_type") or "").lower().strip()
                ename = (item.get("entity_name") or "").strip()
                if not etype or not ename:
                    continue
                key = f"{etype}|{ename.lower()}"
                if key in self.seen:
                    debug_emit({"ev":"item_dedupe_skip", "key": key}, dbg)
                    continue

                debug_emit({"ev":"item_consider", "key": key}, dbg)
                imgs = google_image_search(ename, num=4, dbg=dbg)
                debug_emit({"ev":"item_search_results", "key": key, "n": len(imgs)}, dbg)

                if not imgs:
                    continue

                ok_url = ""
                note = ""
                tries = 0
                for cand in imgs[:2]:
                    tries += 1
                    v = validate_image_with_context(cand["url"], etype, ename, last_q, reply, dbg=dbg)
                    if v.get("ok"):
                        ok_url = cand["url"]
                        note = v.get("reason","")
                        debug_emit({"ev":"item_validate_ok", "key": key, "tries": tries}, dbg)
                        break
                    else:
                        debug_emit({"ev":"item_validate_ko", "key": key, "tries": tries, "reason": v.get("reason","")[:160]}, dbg)

                if not ok_url:
                    debug_emit({"ev":"item_no_valid_image", "key": key}, dbg)
                    continue

                self.seen.append(key)
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
                    "best_image_url": ok_url,
                    "candidates": [{"url": ok_url, "page_url": imgs[0].get("page_url",""), "source": "google-cse", "confidence": 0.9, "reason": note}],
                    "notes": note or "validated"
                }

                sid = next_free_slot()
                if not sid:
                    results.append({"status": "full", "label": label})
                    debug_emit({"ev":"slots_full"}, dbg)
                    break
                results.append({"status": "ok", "sid": sid, "label": label, "media": media})
                debug_emit({"ev":"slot_created", "sid": sid, "label": label}, dbg)
                processed += 1

            return {"status": "batch", "items": results, "debug": dbg}
        except Exception as e:
            debug_emit({"ev":"job_error", "error": str(e)}, dbg)
            return {"status": "error", "error": str(e), "trace": traceback.format_exc()[:1200], "debug": dbg}

    def poll(self) -> List[str]:
        updated, rm = [], []
        for jid, (sid, fut) in list(self.jobs.items()):
            if fut.done():
                rm.append(jid)
                try:
                    res = fut.result()
                except Exception as e:
                    debug_emit({"ev":"poll_exception", "error": str(e)})
                    continue

                # Worker-Debug in Main-Thread-Log pushen
                for ev in res.get("debug", []) or []:
                    debug_emit(ev, buffer=None)

                if res.get("status") == "batch":
                    for it in res.get("items", []):
                        if it.get("status") == "ok":
                            self.upsert(it["sid"], it["label"], it["media"])
                            updated.append(it["sid"])
                elif res.get("status") == "skip":
                    debug_emit({"ev":"poll_skip"})
                elif res.get("status") == "error":
                    debug_emit({"ev":"poll_error", "error": res.get("error")})
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

def render_debug_sidebar():
    st.sidebar.header("🔍 Agent 2 Debug")
    st.sidebar.caption("Extraktion → Google CSE → Vision → Slots")
    st.sidebar.write(f"Google CSE key: {'✅ set' if bool(GOOGLE_CSE_KEY) else '❌ MISSING'}")
    st.sidebar.write(f"Google CSE cx : {'✅ set' if bool(GOOGLE_CSE_CX)  else '❌ MISSING'}")
    st.sidebar.write(f"Seen entities: {len(st.session_state.seen_entities)}")
    if st.session_state.debug_agent2:
        for entry in reversed(st.session_state.debug_agent2):
            st.sidebar.json(entry, expanded=False)
    else:
        st.sidebar.caption("No Agent-2 events yet.")

# ---------- Main
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
st.caption("Agent 2: Google Image API (+Vision-Check) • Sichtbares Debug • Slots nur bei validem Bild")

init_state()

orch = Orchestrator()

# First opener from Agent 1
if not st.session_state.history:
    opener = agent1_next_question([])
    st.session_state.history.append({"role": "assistant", "content": opener})
    st.rerun()

# Poll Agent 2 async jobs
for sid in orch.poll():
    st.toast(f"🖼️ Media updated: {sid}", icon="🖼️")

# Render UI
render_slots()
render_history()
render_debug_sidebar()

# Auto-finalize fallback
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

    # Agent 2: extract → google image → vision
    orch.schedule_watch(last_q, user_text)

    # Agent 1: next question
    nxt = agent1_next_question(st.session_state.history)
    st.session_state.history.append({"role": "assistant", "content": nxt})

    # Optional Handoff Phrase
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
