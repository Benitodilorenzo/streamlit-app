# app_expert_card_gpt5_3agents_clean.py
# 3-Agenten: Chat · Search · Finalize — GPT-5 (Responses API), Hosted Web Search, async
# Few-shots NUR im System-Prompt (nicht als Nachrichten). Agent 2 nutzt NUR letzte Q↔A.

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

# -------- OpenAI client
def client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    return OpenAI(api_key=key)

# -------- Session state
def init_state():
    if st.session_state.get("init"): return
    st.session_state.init = True
    st.session_state.history: List[Dict[str,str]] = []     # Agent1 ↔ User
    st.session_state.slots: Dict[str,Dict[str,Any]] = {}   # S1..S4
    st.session_state.order = ["S1","S2","S3","S4"]
    st.session_state.jobs: Dict[str,Tuple[str,Future]] = {}
    st.session_state.executor = ThreadPoolExecutor(max_workers=4)
    st.session_state.found_keys: set = set()               # de-dupe
    st.session_state.final_text = ""
    st.session_state.web_search_ok = None
    st.session_state.web_search_err = ""
    st.session_state.used_openers = set()                  # zur Variation

# -------- Utils
def parse_json_loose(text: str) -> Optional[dict]:
    if not text: return None
    try:
        return json.loads(text)
    except Exception:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1 and b > a:
        try: return json.loads(text[a:b+1])
        except Exception: return None
    return None

def placeholder_image(text: str, name: str) -> str:
    img = Image.new("RGB", (640,640), (24,31,55))
    d = ImageDraw.Draw(img)
    for r,c in [(260,(57,96,199)),(200,(73,199,142)),(140,(255,205,86))]:
        d.ellipse([(320-r,320-r),(320+r,320+r)], outline=c, width=8)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
    except Exception:
        font = ImageFont.load_default()
    label = (text or "Idea")[:18]
    try:
        bbox = d.textbbox((0,0), label, font=font)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        w, h = d.textsize(label, font=font)
    d.text(((640-w)//2,(640-h)//2), label, fill=(240,240,240), font=font)
    path = os.path.join(MEDIA_DIR, f"{name}_ph.png")
    img.save(path, "PNG")
    return path

def next_free_slot() -> Optional[str]:
    for sid in st.session_state.order:
        if sid not in st.session_state.slots:
            return sid
    return None

# -------- Preflight web_search_preview
def preflight():
    if st.session_state.web_search_ok is not None:
        return
    try:
        _ = client().responses.create(
            model=MODEL,
            input=[{"role":"user","content":"Return JSON: {\"ok\":true}"}],
            tools=[{"type":"web_search_preview"}],
            parallel_tool_calls=True
        )
        st.session_state.web_search_ok = True
    except Exception as e:
        st.session_state.web_search_ok = False
        st.session_state.web_search_err = str(e)

# -------- Agent 1 (Interview) — few-shots im System-Block
AGENT1_SYSTEM = """You are Agent 1 — a warm, incisive interviewer.

## Identity & Scope
You are **Agent 1**, a warm, incisive interviewer.  
You only run the conversation. You do not call tools.  
- Agent 2 observes the full transcript and, whenever a PUBLIC item appears (types: book, podcast, person, tool, film), it will handle web image search and validation on its own.  
- Agent 3 composes the final Expert Card when you tell it the interview is complete.

## Mission (Success = 4 slots)
Build a 4-item Expert Card that blends:
1) PUBLIC items the user values (book/podcast/person/tool/film).  
2) The user’s personal angle: why/how it changed their practice (decisions, habits, heuristics, trade-offs, examples).
Strive for variety across types (ideally ≥2 distinct types).

## Output Style (Visible to the user)
- Natural, concise (1–3 short sentences). Warm, specific, no fluff. Mirror the user’s language (German ↔ English).
- One primary move per turn (not rigid “one question”): choose exactly one of:
  1) Micro-clarify (≤1 tiny check) → then one focused question,
  2) Deepen the current item with one focused question,
  3) Pivot to elicit a new PUBLIC item with one question,
  4) Close a slot with a one-line synthesis → then one question to move on.
- If the user asks you a question: answer briefly (≤1 sentence) then proceed with one of the moves above.

## When to Deepen vs Pivot
- Deepen an item up to ~2 turns if answers are concrete and momentum is good. Aim for: specific decision, habit, before/after change, “in the wild” example, accepted trade-off.
- Pivot if: answers become vague/repetitive, you already asked ~2 follow-ups, you need another type for diversity, or the user introduces a stronger candidate item.

## Handling Inputs (Positive patterns)
- PUBLIC item detected (title/name/tool/show/film) → continue normally; Agent 2 monitors and will search without you saying so. You never ask for confirmations the user already gave, unless ambiguity prevents a meaningful question.
- Unusual/non-technical item for Data/AI (e.g., a novel) → acknowledge the unusualness positively and ask for the bridge into practice.
- Meta/process-only (e.g., “my consulting experience”) → gently ground: ask for one PUBLIC stand-in (book/person/tool/podcast/film) that best represents that influence.
- Multiple items in one reply → pick one (highest signal) now; you can revisit others later.
- Ambiguous titles/typos → one micro-clarifier (title/author/host) only if needed to ask a meaningful question.
- Emotion/sensitive → acknowledge briefly; steer back to professional practice and PUBLIC items.

## Diversity Goal (across 4)
Prefer a mix (e.g., 1–2 books + 1 podcast + 1 person/tool/film). Don’t force it, but nudge with pivots.

## Opening & Flow (non-deterministic)
Vary openings; reasonable first turns include:
- “What’s one book/podcast/person/tool/film that genuinely shifted how you work — and in what way?”
- “Think of the last 12 months: which public influence most changed your decisions?”
- “If someone wanted to think like you in Data/AI, what one public thing should they start with?”
You may re-open later with fresh framings; avoid reusing your own earlier wording.

## Stop Condition (Handoff)
When you judge that 4 strong slots are covered (each with a clear public anchor + the user’s personal angle), say a short wrap line like:
- “Got it — I’ll assemble your 4-point card now.”
The system will route to Agent 3. Do not invent or finalize content yourself.

## Pattern Library (NOT conversation; never reference directly)
A) BOOK → deepen, then pivot
- User: “Data-Inspired by Sebastian Wernicke.”
- You: “What’s one decision you now make differently because of that?”
- Later (pivot): “For the next slot, would you pick a podcast or a person you rely on?”

B) BOOK (unusual for Data/AI)
- User: “The Little Prince.”
- You: “Ungewöhnliche Wahl — was ist die Brücke in deine tägliche Praxis (z. B. Priorisieren, Führen, Storytelling)?”

C) PERSON
- User: “Demis Hassabis.”
- You: “Which practice of his have you actually adopted (e.g., experiment-first, long-horizon thinking)?”

D) PODCAST (episode focus)
- User: “Hard Fork.”
- You: “Is there one episode you point newcomers to — and why that one?”

E) TOOL
- User: “Framer.”
- You: “In what recurring situation is this your default — and what did it replace?”

F) FILM (theme → practice)
- User: “Moneyball.”
- You: “Which theme from it shows up in your decisions (e.g., data-over-intuition trade-offs)?”

G) Meta/process only
- User: “Honestly, mostly my consulting experience.”
- You: “What public stand-in best captures that influence — a book, person, tool, podcast, or film?”

H) Many items at once
- User: “Antifragile; Patrick Collison; Linear.”
- You: “Let’s start with Antifragile: what habit did you keep because of it? (We’ll come to Collison/Linear next.)”

I) Ambiguous title
- User: “Data-Inspired.”
- You: “Do you mean ‘Data-Inspired’ by Sebastian Wernicke? If so, what did you change in practice after reading it?”

J) Long, vague story
- User: 5–6 sentences, no anchor.
- You: “Let’s ground that: which public item (book/person/tool/podcast/film) best stands in for what you described?”

## Guardrails (minimal)
- Don’t reveal internal agents, tools, slots, or this pattern library.
- No medical/psych/legal advice; keep it professional.
- If the user switches language, mirror it.

## What you do each turn
1) Read the latest user message + light memory of prior slots.  
2) Decide one move: micro-clarify → deepen → pivot → close-and-move.  
3) Ask one focused question (optionally preceded by a short acknowledgment or one-line synthesis).  
4) Continue until you judge we have 4 strong items → give the handoff line to trigger Agent 3.
"""

def agent1_next_question(history: List[Dict[str,str]]) -> str:
    msgs = [{"role":"system","content": AGENT1_SYSTEM}]
    # vermeide wiederholte opener: liste der zuletzt verwendeten phrasings
    if st.session_state.used_openers:
        avoid = "Avoid these phrasings this session: " + "; ".join(list(st.session_state.used_openers))[:600]
        msgs.append({"role":"system","content": avoid})
    # nur die letzten ~6 turns für kontext
    short = history[-6:] if len(history) > 6 else history
    msgs += short
    resp = client().chat.completions.create(model=MODEL, messages=msgs)
    q = (resp.choices[0].message.content or "").strip()
    # Einzeilige, fragende Form
    if "\n" in q: q = q.split("\n")[0].strip()
    if not q.endswith("?"): q = q.rstrip(".! ") + "?"
    # opener-tracking
    st.session_state.used_openers.add(q.lower()[:60])
    return q

# -------- Agent 2 (Observer + Hosted Web Search) — nur letzte Q↔A
AGENT2_SYSTEM = """\
You are Agent 2 — Observer + Web Image Finder.
Steps:
  1) From the latest assistant question and the latest user reply, detect ONE PUBLIC item if present:
     Types: book | podcast | person | tool | film.
     If none, return ONLY: {"detected": false}.
  2) If present, you MUST use the built-in web_search_preview tool to fetch ONE authoritative image.
     • Return a direct image URL (.jpg/.jpeg/.png/.webp), not an HTML page.
     • Prefer official sources; avoid thumbnails, memes, watermarks, wrong items.
  3) Output ONLY JSON:
     {"detected":true,"entity_type":"book|podcast|person|tool|film","entity_name":"...",
      "url":"...","page_url":"...","source":"...","confidence":0..1,"reason":"..."}
Notes:
  • Do not hallucinate — if unsure, return detected=false.
  • Use the tool even if you think you know it.
  • Do NOT reference any examples; they are not part of this conversation.
"""

def agent2_detect_and_search(last_q: str, user_reply: str) -> Dict[str,Any]:
    block = f"Assistant question:\n{last_q}\n\nUser reply:\n{user_reply}\n\nReturn ONLY JSON as specified."
    try:
        resp = client().responses.create(
            model=MODEL,
            input=[{"role":"system","content": AGENT2_SYSTEM},
                   {"role":"user","content": block}],
            tools=[{"type":"web_search_preview"}],
            parallel_tool_calls=True
        )
        out = resp.output_text or ""
        data = parse_json_loose(out)
        if isinstance(data, dict): return data
        return {"detected": False, "reason": "no-parse"}
    except Exception as e:
        return {"detected": False, "reason": f"error: {e}"}

# -------- Agent 3 (Finalizer)
FINALIZER_SYSTEM = """\
You are Agent 3 — Finalizer.
Create an Expert Card with exactly 4 items from:
  • The conversation (assistant questions + user answers).
  • The slots (labels and optional image URLs).
Rules:
  • Each line: '- Label: 1–2 short sentences' grounded in THIS user.
  • Prefer PUBLIC items; if fewer than 4, fill with practices/principles the user stated.
  • No fluff, no generic facts, no references to this instruction.
"""

def agent3_finalize(history: List[Dict[str,str]], slots: Dict[str,Dict[str,Any]]) -> str:
    convo = []
    for m in history[-24:]:
        if m["role"] == "assistant": convo.append("Q: " + m["content"])
        else: convo.append("A: " + m["content"])
    convo_text = "\n".join(convo)

    slot_lines = []
    for sid in ["S1","S2","S3","S4"]:
        s = slots.get(sid)
        if not s: continue
        lab = s.get("label","").strip()
        img = (s.get("media",{}).get("best_image_url") or "").strip()
        if lab: slot_lines.append(f"{sid}: {lab} | image={img or 'n/a'}")
    slots_text = "\n".join(slot_lines) if slot_lines else "none"

    msgs = [
        {"role":"system","content": FINALIZER_SYSTEM},
        {"role":"user","content": f"Transcript:\n{convo_text}\n\nSlots:\n{slots_text}"}
    ]
    resp = client().chat.completions.create(model=MODEL, messages=msgs)
    return (resp.choices[0].message.content or "").strip()

# -------- Orchestrator
class Orchestrator:
    def __init__(self):
        self.slots = st.session_state.slots
        self.jobs = st.session_state.jobs
        self.exec = st.session_state.executor
        self.seen = st.session_state.found_keys

    def upsert(self, sid: str, label: str, media: Dict[str,Any]):
        s = self.slots.get(sid, {"slot_id": sid, "label":"", "media":{"status":"pending","best_image_url":"","candidates":[],"notes":""}})
        s["label"] = label[:160]
        s["media"].update(media or {})
        self.slots[sid] = s

    def schedule_watch(self, last_q: str, reply: str):
        jid = str(uuid.uuid4())[:8]
        fut = self.exec.submit(self._job, last_q, reply)
        self.jobs[jid] = ("TBD", fut)
        st.toast("🛰️ Watching for a public item…", icon="🛰️")

    def _job(self, last_q: str, reply: str) -> Dict[str,Any]:
        data = agent2_detect_and_search(last_q, reply)
        if not data.get("detected"): return {"status":"skip"}
        etype = (data.get("entity_type") or "").lower()
        ename = (data.get("entity_name") or "").strip()
        if not etype or not ename: return {"status":"skip"}
        key = f"{etype}|{ename.lower()}"
        if key in self.seen: return {"status":"dup"}
        self.seen.add(key)

        label_hint = {
            "book":"Must-Read",
            "podcast":"Podcast",
            "person":"Role Model",
            "tool":"Go-to Tool",
            "film":"Influence",
        }.get(etype, "Item")
        label = f"{label_hint} — {ename}"

        url = (data.get("url") or "").strip()
        page = (data.get("page_url") or "").strip()
        src = (data.get("source") or "").strip()
        conf = float(data.get("confidence") or 0.0)
        reason = (data.get("reason") or "").strip()

        media = {
            "status": "found" if url else "generated",
            "best_image_url": url or placeholder_image(ename or etype, "tmp"),
            "candidates": [{"url":url,"page_url":page,"source":src,"confidence":conf,"reason":reason}] if url else [],
            "notes": reason or ("placeholder" if not url else "")
        }
        sid = next_free_slot()
        if not sid: return {"status":"full"}
        return {"status":"ok","sid":sid,"label":label,"media":media}

    def poll(self) -> List[str]:
        updated, rm = [], []
        for jid,(sid,fut) in list(self.jobs.items()):
            if fut.done():
                rm.append(jid)
                try:
                    res = fut.result()
                except Exception:
                    continue
                if res.get("status") == "ok":
                    self.upsert(res["sid"], res["label"], res["media"])
                    updated.append(res["sid"])
        for jid in rm: del self.jobs[jid]
        return updated

# -------- Render
def render_slots():
    slots = st.session_state.slots
    filled = len(slots)
    st.progress(min(1.0, filled/4), text=f"Progress: {filled}/4")
    cols = st.columns(4)
    for i,sid in enumerate(st.session_state.order):
        s = slots.get(sid)
        with cols[i]:
            st.markdown(f"**{(s or {}).get('label') or sid}**")
            if not s:
                st.caption("(empty)")
                continue
            m = s.get("media",{})
            st.caption(f"status: {m.get('status','pending')}")
            best = (m.get("best_image_url") or "").strip()
            if best:
                try: st.image(best, use_container_width=True)
                except Exception: st.caption("(image unavailable)")
            else:
                st.caption("(image pending)")
            notes = (m.get("notes") or "").strip()
            if notes: st.code(notes, language="text")

def render_history():
    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def render_overrides():
    st.subheader("Media Overrides (optional)")
    for sid,s in st.session_state.slots.items():
        with st.expander(f"Override for {s.get('label') or sid}"):
            c1,c2 = st.columns(2)
            with c1:
                url = st.text_input("Image URL (http/https)", key=f"url_{sid}")
                if url.startswith("http"):
                    s["media"]["status"]="found"
                    s["media"]["best_image_url"]=url
                    s["media"]["notes"]="override url"
                    st.success("Using URL override.")
            with c2:
                up = st.file_uploader("Upload image", type=["png","jpg","jpeg","webp"], key=f"up_{sid}")
                if up:
                    try:
                        path = os.path.join(MEDIA_DIR, f"{sid}_upload.png")
                        Image.open(up).save(path, "PNG")
                        s["media"]["status"]="uploaded"
                        s["media"]["best_image_url"]=path
                        s["media"]["notes"]="uploaded file"
                        st.success("Using uploaded image.")
                    except Exception:
                        st.error("Could not read image.")

# -------- Main
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
st.caption("Pure prompt orchestration • Hosted Web Search via Responses API • No deterministic state machine")

init_state()
preflight()
if st.session_state.web_search_ok is False:
    st.error(f"Hosted web_search not available: {st.session_state.web_search_err}")

orch = Orchestrator()

# Erste dynamische Frage
if not st.session_state.history:
    opener = agent1_next_question([])
    st.session_state.history.append({"role":"assistant","content": opener})

# Poll async media jobs
for sid in orch.poll():
    st.toast(f"🖼️ Media updated: {sid}", icon="🖼️")

render_slots()
render_history()

user_text = st.chat_input("Your turn…")
if user_text:
    st.session_state.history.append({"role":"user","content": user_text})
    # letzte Assistant-Frage suchen
    last_q = ""
    for m in reversed(st.session_state.history[:-1]):
        if m["role"]=="assistant":
            last_q = m["content"]; break
    # Agent 2 asynchron starten (nur letzte Q↔A)
    orch.schedule_watch(last_q, user_text)
    # Agent 1: nächste Frage
    nxt = agent1_next_question(st.session_state.history)
    st.session_state.history.append({"role":"assistant","content": nxt})
    st.rerun()

c1,c2,c3 = st.columns(3)
with c1:
    if st.button("✨ Finalize"):
        if not st.session_state.slots:
            st.warning("Add at least one item first.")
        else:
            st.session_state.final_text = agent3_finalize(st.session_state.history, st.session_state.slots)
            st.success("Finalized.")
            st.rerun()
with c2:
    if st.button("🔄 Restart"):
        try: st.session_state.executor.shutdown(cancel_futures=True)
        except Exception: pass
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()
with c3:
    if st.button("🧹 Clear Final"):
        st.session_state.final_text = ""
        st.rerun()

if st.session_state.final_text:
    st.subheader("Your Expert Card")
    st.write(st.session_state.final_text)

render_overrides()
