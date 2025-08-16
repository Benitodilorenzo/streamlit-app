# app_expert_card_gpt5_3agents_clean.py
# 3 Agents: Chat (Agent 1) · Search (Agent 2) · Finalize (Agent 3)
# GPT-5 (Responses API) with hosted web_search_preview; async Agent 2 jobs.
# Agent 2 processes ONLY the last Q↔A + seen_entities, extracts 0..N items,
# de-dupes, and searches up to MAX_SEARCHES per turn. Orchestrator maps to slots.
#
# Requirements:
#   pip install streamlit openai pillow
#
# Env:
#   OPENAI_API_KEY              (required)
#   OPENAI_GPT5_SNAPSHOT        (optional, default: gpt-5-2025-08-07)

import os, json, uuid
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

APP_TITLE = "🟡 Expert Card — GPT-5 (3 Agents · Hosted Web Search · Async)"
MODEL = os.getenv("OPENAI_GPT5_SNAPSHOT", "gpt-5-2025-08-07")
MEDIA_DIR = os.path.join(os.getcwd(), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

MAX_SEARCHES_PER_RUN = 2  # Agent 2: cap per turn

# ---------- OpenAI client
def client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    return OpenAI(api_key=key)

# ---------- Session state
def init_state():
    if st.session_state.get("init"):
        return
    st.session_state.init = True
    st.session_state.history: List[Dict[str, str]] = []          # Agent1 ↔ User
    st.session_state.slots: Dict[str, Dict[str, Any]] = {}       # S1..S4
    st.session_state.order = ["S1", "S2", "S3", "S4"]
    st.session_state.jobs: Dict[str, Tuple[str, Future]] = {}     # Agent 2 futures
    st.session_state.executor = ThreadPoolExecutor(max_workers=4)
    st.session_state.seen_entities: List[str] = []               # e.g., "book|the little prince"
    st.session_state.final_text = ""
    st.session_state.web_search_ok = None
    st.session_state.web_search_err = ""
    st.session_state.used_openers = set()                        # Agent 1 self-variation
    st.session_state.auto_finalized = False

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
        try:
            return json.loads(text[a:b+1])
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

# ---------- Preflight hosted web_search_preview
def preflight():
    if st.session_state.web_search_ok is not None:
        return
    try:
        _ = client().responses.create(
            model=MODEL,
            input=[{"role": "user", "content": "Return JSON: {\"ok\":true}"}],
            tools=[{"type": "web_search_preview"}],
            parallel_tool_calls=True
        )
        st.session_state.web_search_ok = True
    except Exception as e:
        st.session_state.web_search_ok = False
        st.session_state.web_search_err = str(e)

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
- Unusual/non-technical item → acknowledge positively and ask for the bridge to practice.
- Meta/process-only → ask for one PUBLIC stand-in (book/person/tool/podcast/film).
- Multiple items in one reply → pick one now; you may revisit others later.
- Ambiguous → one micro-clarifier (title/author/host) if needed.
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
    resp = client().chat.completions.create(model=MODEL, messages=msgs)
    q = (resp.choices[0].message.content or "").strip()
    if "\n" in q:
        q = q.split("\n")[0].strip()
    if not q.endswith("?"):
        q = q.rstrip(".! ") + "?"
    st.session_state.used_openers.add(q.lower()[:72])
    return q

# ---------- Agent 2 (Observer + Web Image Finder) — multi-item; de-dupe; capped searches
# ---------- Agent 2 (Observer + Web Image Finder) — multi-item; de-dupe; capped searches
AGENT2_SYSTEM = (
"""You are Agent 2 — Observer + Web Image Finder.

Inputs (provided as a single JSON in the user message each turn):
- assistant_question: last question from Agent 1
- user_reply: latest user reply
- seen_entities: array of strings like "book|the little prince" already processed

Your job each turn:
1) From assistant_question + user_reply, extract ZERO OR MORE PUBLIC items:
   Types: book | podcast | person | tool | film.
   Use the question only for disambiguation; do not invent items.

2) DEDUPE: For each detected item, build a key "type|normalized_name" (lowercase, trim quotes).
   If the key is in seen_entities, DO NOT search it again; mark action="skipped_duplicate".

3) For each new item, use the built-in web_search_preview tool to find ONE authoritative image:
   - Return a direct image URL (.jpg/.jpeg/.png/.webp), not an HTML page.
   - Prefer official sources; avoid thumbnails, memes, watermarks, wrong items.
"""
+ f"   - Perform at most MAX_SEARCHES={MAX_SEARCHES_PER_RUN} tool calls per run.\n"
+ """
   - If more new items exist than the cap, include the extras with action="detected_no_search" (no URL).

4) OUTPUT ONLY strict JSON:
{
  "detected": true|false,
  "items": [
    {
      "entity_type": "book|podcast|person|tool|film",
      "entity_name": "...",
      "action": "searched|skipped_duplicate|detected_no_search",
      "url": "...",
      "page_url": "...",
      "source": "...",
      "confidence": 0.0,
      "reason": "one short sentence"
    }
  ]
}

Rules:
- If no item is present, return {"detected": false}.
- Never duplicate the same item within one response.
- Never fabricate URLs; always use the tool for items you mark as "searched".
- Keep reasons short.
"""
)


def agent2_detect_and_search_multi(last_q: str, user_reply: str, seen_entities: List[str]) -> Dict[str, Any]:
    payload = {
        "assistant_question": last_q or "",
        "user_reply": user_reply or "",
        "seen_entities": list(seen_entities or [])
    }
    try:
        resp = client().responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": AGENT2_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            tools=[{"type": "web_search_preview"}],
            parallel_tool_calls=True
        )
        out = resp.output_text or ""
        data = parse_json_loose(out)
        if isinstance(data, dict):
            return data
        return {"detected": False, "reason": "no-parse"}
    except Exception as e:
        return {"detected": False, "reason": f"error: {e}"}

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
    resp = client().chat.completions.create(model=MODEL, messages=msgs)
    return (resp.choices[0].message.content or "").strip()

# ---------- Orchestrator
class Orchestrator:
    def __init__(self):
        self.slots = st.session_state.slots
        self.jobs = st.session_state.jobs
        self.exec = st.session_state.executor
        self.seen = st.session_state.seen_entities   # list of keys type|name

    def upsert(self, sid: str, label: str, media: Dict[str, Any]):
        s = self.slots.get(sid, {"slot_id": sid, "label": "", "media": {"status": "pending", "best_image_url": "", "candidates": [], "notes": ""}})
        s["label"] = label[:160]
        s["media"].update(media or {})
        self.slots[sid] = s

    def schedule_watch(self, last_q: str, reply: str):
        jid = str(uuid.uuid4())[:8]
        fut = self.exec.submit(self._job, last_q, reply, list(self.seen))
        # We keep "TBD" — multiple slots may be updated; job id only tracks completion.
        self.jobs[jid] = ("TBD", fut)
        st.toast("🛰️ Agent 2 is scanning for public items…", icon="🛰️")

    def _job(self, last_q: str, reply: str, seen_snapshot: List[str]) -> Dict[str, Any]:
        data = agent2_detect_and_search_multi(last_q, reply, seen_snapshot)
        if not data.get("detected"):
            return {"status": "skip"}
        items = data.get("items") or []
        results: List[Dict[str, Any]] = []

        for item in items:
            etype = (item.get("entity_type") or "").lower()
            ename = (item.get("entity_name") or "").strip()
            action = (item.get("action") or "").strip()
            if not etype or not ename:
                continue

            key = f"{etype}|{ename.lower()}"
            if key in self.seen:
                # Even if Agent 2 marked it skipped, ensure dedupe here too
                continue

            # Record as seen (prevents future repeats)
            self.seen.append(key)

            label_hint = {
                "book": "Must-Read",
                "podcast": "Podcast",
                "person": "Role Model",
                "tool": "Go-to Tool",
                "film": "Influence",
            }.get(etype, "Item")
            label = f"{label_hint} — {ename}"

            url = (item.get("url") or "").strip()
            page = (item.get("page_url") or "").strip()
            src = (item.get("source") or "").strip()
            conf = float(item.get("confidence") or 0.0) if "confidence" in item else 0.0
            rsn = (item.get("reason") or "").strip()

            media = {
                "status": "found" if (action == "searched" and url) else ("pending" if action == "detected_no_search" else "skipped"),
                "best_image_url": url or (placeholder_image(ename or etype, "tmp") if action != "skipped_duplicate" else ""),
                "candidates": ([{"url": url, "page_url": page, "source": src, "confidence": conf, "reason": rsn}] if url else []),
                "notes": rsn or (action if action else "")
            }

            sid = next_free_slot()
            if not sid:
                results.append({"status": "full", "label": label})
                break
            results.append({"status": "ok", "sid": sid, "label": label, "media": media})

        return {"status": "batch", "items": results}

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
                # skip/other statuses are ignored
        for jid in rm:
            del self.jobs[jid]
        return updated

# ---------- Render
def render_slots():
    slots = st.session_state.slots
    filled = len([s for s in slots.values() if s.get("label")])
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
st.caption("Pure prompt orchestration • Hosted Web Search via Responses API • Agent 2 does multi-item extraction + de-dupe per turn")

init_state()
preflight()
if st.session_state.web_search_ok is False:
    st.error(f"Hosted web_search not available: {st.session_state.web_search_err}")

orch = Orchestrator()

# First dynamic opener from Agent 1
if not st.session_state.history:
    opener = agent1_next_question([])
    st.session_state.history.append({"role": "assistant", "content": opener})

# Poll Agent 2 async jobs
updated = orch.poll()
for sid in updated:
    st.toast(f"🖼️ Media updated: {sid}", icon="🖼️")

# Render UI
render_slots()
render_history()

# Auto-finalize fallback: if 4 slots filled and not yet finalized, run Agent 3
if not st.session_state.auto_finalized:
    if all(sid in st.session_state.slots and st.session_state.slots[sid].get("label") for sid in st.session_state.order):
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

    # Agent 2: monitor + search (only last Q↔A), pass seen_entities for dedupe
    orch.schedule_watch(last_q, user_text)

    # Agent 1: next question
    nxt = agent1_next_question(st.session_state.history)
    st.session_state.history.append({"role": "assistant", "content": nxt})

    # If Agent 1 signals handoff, finalize now (even if <4, it'll fill with practices)
    handoff_trigger = "assemble your 4" in nxt.lower() or "assemble your 4-point" in nxt.lower() or "assemble your 4-point" in nxt.lower()
    if handoff_trigger:
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
