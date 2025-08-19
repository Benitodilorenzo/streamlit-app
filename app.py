# app.py — 🟡 Expert Card – 3-Agent System (Single File, Responses API correct)
# Requirements: streamlit, openai>=1.40, requests
# ENV: OPENAI_API_KEY, GOOGLE_CSE_KEY, GOOGLE_CSE_CX, GOOGLE_CSE_SAFE (off|active)

import os, json, uuid, requests
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI
from requests.exceptions import HTTPError

# -------------------------
# Config / ENV
# -------------------------
APP_TITLE = "🟡 Expert Card — GPT-5 (3 Agents · Responses API)"
MODEL = os.getenv("OPENAI_GPT5_SNAPSHOT", "gpt-5")           # API alias per GPT-5 guide :contentReference[oaicite:1]{index=1}
AGENT2_MODEL = os.getenv("OPENAI_AGENT2_MODEL", "gpt-5-mini") # cost-optimized validator :contentReference[oaicite:2]{index=2}

GOOGLE_CSE_KEY  = os.getenv("GOOGLE_CSE_KEY", "").strip()
GOOGLE_CSE_CX   = os.getenv("GOOGLE_CSE_CX", "").strip()
GOOGLE_CSE_SAFE = os.getenv("GOOGLE_CSE_SAFE", "off").strip().lower()  # "off" | "active"

HANDOFF_PHRASES = [
    "i’ll assemble your 4-point card now",
    "i'll assemble your 4-point card now",
    "we have 4 aspects we can turn into a profile",
    "we have four aspects we can turn into a profile",
    "wir haben nun 4 aspekte",
    "wir haben jetzt 4 aspekte",
    "wir haben vier aspekte",
    "ich stelle jetzt deinen steckbrief zusammen",
    "ich erstelle jetzt deinen 4-punkte-steckbrief",
    "alright — i’ll now assemble your 4-point card",
    "alright - i’ll now assemble your 4-point card",
    "alright — i'll now assemble your 4-point card",
    "alright - i'll now assemble your 4-point card",
    "perfect — i’ll now assemble your selected 4-point card",
    "perfect — i'll now assemble your selected 4-point card",
    "great — i’ll now assemble your 4-point expert card with your chosen items",
    "great — i'll now assemble your 4-point expert card with your chosen items",
]

# -------------------------
# OpenAI client
# -------------------------
def client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        st.error("OPENAI_API_KEY fehlt.")
        st.stop()
    return OpenAI(api_key=key)

# -------------------------
# Schemas (Structured Outputs via text.format json_schema)
# -------------------------
ORCHESTRATOR_TURN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "assistant_message": {
            "type": "string",
            "description": "Next assistant utterance (1–3 sentences). If not handing off, end with a question."
        },
        "planned_searches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string", "enum": ["person","book","podcast","tool","film"]},
                    "entity_name": {"type": "string"},
                    "artifact": {"type": "string", "enum": ["portrait","book_cover","podcast_cover","tool_logo","film_poster"]},
                    "search_query": {"type": "string"},
                    "cluster_id": {"type": "string"}
                },
                "required": ["entity_type","entity_name","artifact","search_query","cluster_id"],
                "additionalProperties": False
            }
        }
    },
    "required": ["assistant_message","planned_searches"],
    "additionalProperties": False
}

VALIDATOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "reason": {"type": "string"}
    },
    "required": ["ok","reason"],
    "additionalProperties": False
}

EXPERT_CARD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "string"}
        },
        "html": {"type": "string", "description": "Exportable HTML snippet for the 4 items with images."}
    },
    "required": ["lines","html"],
    "additionalProperties": False
}

# -------------------------
# System prompts (compact; du kannst deine großen später einsetzen)
# -------------------------
SYSTEM_AGENT1: str = (
    "You are Agent 1 — interviewer/orchestrator.\n"
    "Goal: build a 4-item Expert Card. Keep turns short (1–3 sentences). One move per turn: clarify, deepen, pivot, or close.\n"
    "Detect public items (book/podcast/person/tool/film) and plan precise image searches (artifact-aware; add disambiguation like “<Title> by <Author> book cover”).\n"
    "Offer to assemble after 4 solid anchors; only hand off after explicit confirmation.\n"
    "OUTPUT strictly matches the JSON schema: assistant_message + planned_searches.\n"
    "Do not mention tools."
)

SYSTEM_AGENT1_CHAT: str = (
    "You are Agent 1 — the user-facing voice. Speak naturally and briefly. "
    "Don’t mention tools or planning. If finalizing, use one of the handoff phrases exactly; else ask one focused question."
)

SYSTEM_VALIDATOR: str = (
    "You are Agent 2 — image validator. STRICT JSON. For PERSON+portrait: DO NOT identify the person; only check portrait form (single human, not logo/meme/group). "
    "For covers/posters/logos: check plausibility for the requested artifact given the context. Return {ok:boolean, reason:string}."
)

SYSTEM_FINALIZER: str = (
    "You are Agent 3 — Finalizer. Produce exactly 4 lines (concise, grounded in this user's words). "
    "Also produce an HTML snippet that lays out the 4 items with optional images (title on left with 1–3 sentences, image on right)."
)

# -------------------------
# Helpers
# -------------------------
def _resp_text(resp) -> str:
    try:
        t = getattr(resp, "output_text", None)
        if t:
            return t
    except Exception:
        pass
    try:
        out = resp.output or []
        buf = []
        for item in out:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") in ("output_text","text"):
                        buf.append(getattr(c, "text", "") or "")
        return "".join(buf)
    except Exception:
        return ""

def safe_json(txt: str) -> Dict[str, Any]:
    try:
        return json.loads(txt)
    except Exception:
        a, b = txt.find("{"), txt.rfind("}")
        if a != -1 and b != -1 and b > a:
            try:
                return json.loads(txt[a:b+1])
            except Exception:
                return {}
        return {}

def next_free_slot(slots: Dict[str, Any]) -> Optional[str]:
    for sid in ("S1","S2","S3","S4"):
        if sid not in slots or not (slots.get(sid,{}).get("label")):
            return sid
    return None

def render_slots(slots: Dict[str, Any]):
    filled = len([s for s in slots.values() if (s.get("label") or "").strip()])
    st.progress(min(1.0, filled/4), text=f"Progress: {filled}/4")
    if slots:
        cols = st.columns(4)
        order = ["S1","S2","S3","S4"]
        for idx, sid in enumerate(order):
            with cols[idx]:
                s = slots.get(sid)
                if not s:
                    st.caption(sid); st.write("—"); continue
                st.caption(sid + " · " + s.get("label",""))
                img = (s.get("image_url") or "").strip()
                if img: st.image(img, use_column_width=True)
                else:   st.write("(no image)")

# -------------------------
# Google Image Search (CSE)
# -------------------------
def google_image_search(query: str, num: int = 6) -> List[Dict[str, str]]:
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        return []
    params = {
        "q": query,
        "searchType": "image",
        "num": max(1, min(num, 10)),
        "safe": GOOGLE_CSE_SAFE,
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_CX,
    }
    try:
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", []) or []
        out = []
        for it in items:
            link = (it.get("link") or "").strip()
            ctx = (it.get("image", {}) or {}).get("contextLink") or ""
            if link:
                out.append({"url": link, "page_url": ctx})
        return out
    except HTTPError:
        return []
    except Exception:
        return []

# -------------------------
# Agent calls
# -------------------------
def agent1_plan_searches(last_q: str, user_reply: str, known_labels: List[str]) -> Dict[str, Any]:
    """Structured output: assistant_message + planned_searches[]"""
    c = client()
    payload = {
        "assistant_question": last_q or "",
        "user_reply": user_reply or "",
        "known_slots": known_labels or [],
    }
    resp = c.responses.create(
        model=MODEL,
        text={
            "verbosity": "low",                                    # GPT-5 verbosity :contentReference[oaicite:3]{index=3}
            "format": {
                "type": "json_schema",
                "name": "OrchestratorTurn",
                "schema": ORCHESTRATOR_TURN_SCHEMA,               # ← WICHTIG: schema gesetzt (nicht json_schema)
                "strict": True
            }
        },
        instructions=SYSTEM_AGENT1,
        input=[{"role": "user", "content": [{"type":"input_text","text": json.dumps(payload, ensure_ascii=False)}]}],
    )
    return safe_json(_resp_text(resp))

def agent1_chat_stream(history: List[Dict[str, str]], mode_hint: str, slot_summary: str) -> str:
    """User-visible utterance (streamed)."""
    c = client()
    # small transcript
    convo = []
    for m in history[-10:]:
        prefix = "Q: " if m["role"]=="assistant" else "A: "
        convo.append(prefix + m["content"])
    transcript = "\n".join(convo)

    user_payload = {"transcript": transcript, "mode": mode_hint, "slot_summary": slot_summary}
    container = st.empty()
    buf = ""

    try:
        with c.responses.stream(
            model=MODEL,
            text={"verbosity": "low"},
            instructions=SYSTEM_AGENT1_CHAT,
            input=[{"role":"user","content":[{"type":"input_text","text": json.dumps(user_payload, ensure_ascii=False)}]}]
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if delta:
                        buf += delta
                        container.markdown(buf)
                elif event.type == "response.error":
                    container.error(str(getattr(event, "error", "")))
            final = stream.get_final_response()
            if not buf:
                t = _resp_text(final)
                if t:
                    buf = t
                    container.markdown(buf)
    except Exception as e:
        container.error(f"Stream error: {e}")
    return (buf or "").strip()

def agent2_validate_image(image_url: str, entity_type: str, entity_name: str, artifact: str,
                          q_text: str, a_text: str) -> Dict[str, Any]:
    c = client()
    if (entity_type or "").strip().lower() == "person" and (artifact or "").strip().lower() == "portrait":
        user_text = (
            "Artifact: portrait\nItem type: person\n"
            f"Person name (context only; do NOT identify): {entity_name}\n"
            "Check only portrait form: single human headshot/portrait; not logo/meme/group.\n"
            "Return STRICT JSON."
        )
    else:
        user_text = (
            f"Artifact: {artifact}\nItem type: {entity_type}\nItem name: {entity_name}\n"
            f"Context Q: {q_text[:300]}\nContext A: {a_text[:500]}\n"
            "Check plausibility for requested item/artifact. Return STRICT JSON."
        )
    resp = c.responses.create(
        model=AGENT2_MODEL,
        text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "validator",
                "schema": VALIDATOR_SCHEMA,
                "strict": True
            }
        },
        instructions=SYSTEM_VALIDATOR,
        input=[{"role": "user", "content": [
            {"type":"input_text","text": user_text},
            {"type":"input_image","image_url": image_url}
        ]}],
    )
    return safe_json(_resp_text(resp)) or {"ok": False, "reason": "unverifiable"}

def agent3_finalize(history: List[Dict[str, str]], slots: Dict[str, Any]) -> Dict[str, Any]:
    c = client()
    convo = []
    for m in history[-24:]:
        prefix = "Q: " if m["role"]=="assistant" else "A: "
        convo.append(prefix + m["content"])
    transcript = "\n".join(convo)
    slot_lines = []
    for sid in ["S1","S2","S3","S4"]:
        s = slots.get(sid, {})
        lab = (s.get("label") or "").strip()
        img = (s.get("image_url") or "").strip()
        if lab:
            slot_lines.append(f"{sid}: {lab} | image={img or 'n/a'}")
    user_payload = {"transcript": transcript, "slots": slot_lines}

    resp = c.responses.create(
        model=MODEL,
        text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "expert_card",
                "schema": EXPERT_CARD_SCHEMA,
                "strict": True
            }
        },
        instructions=SYSTEM_FINALIZER,
        input=[{"role":"user","content":[{"type":"input_text","text": json.dumps(user_payload, ensure_ascii=False)}]}],
    )
    return safe_json(_resp_text(resp))

# -------------------------
# Streamlit UI + Orchestration
# -------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
st.caption("Agent-1 (streamed), Agent-2 (validator via image+json_schema), Agent-3 (finalizer via json_schema).")

# Session state
if "started" not in st.session_state:
    st.session_state.started = False
if "mode" not in st.session_state:
    st.session_state.mode = "Professional"
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str,str]] = []
if "slots" not in st.session_state:
    st.session_state.slots: Dict[str,Dict[str,Any]] = {}
if "final_data" not in st.session_state:
    st.session_state.final_data: Dict[str,Any] = {}

# Start gate
st.info("Klicke **Start**, um das Interview zu beginnen. Bis dahin werden **keine** Modell-Aufrufe ausgeführt.")
c1, c2 = st.columns([1,5])
with c1:
    if st.button("▶️ Start", disabled=st.session_state.started):
        st.session_state.started = True
        st.rerun()

if not st.session_state.started:
    st.stop()

# Mode selection
st.session_state.mode = st.radio(
    "Interview focus",
    ["Professional", "General / Lifescope"],
    index=0,
    horizontal=True
)

# Render slots + history
render_slots(st.session_state.slots)
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# First opener (once, after Start)
if not st.session_state.history:
    with st.chat_message("assistant"):
        opener = agent1_chat_stream([], st.session_state.mode, "No items yet.")
    st.session_state.history.append({"role":"assistant", "content": opener if opener else "What’s one book, podcast, person, tool, or film that truly changed how you work — and how?"})
    st.rerun()

# Input
user_text = st.chat_input("Your turn…")
if user_text:
    st.session_state.history.append({"role":"user","content": user_text})

    # PLAN searches (structured)
    last_q = ""
    for m in reversed(st.session_state.history[:-1]):
        if m["role"] == "assistant":
            last_q = m["content"]; break
    known_labels = [v.get("label","") for v in st.session_state.slots.values() if v.get("label")]
    plan = agent1_plan_searches(last_q, user_text, known_labels)

    planned = plan.get("planned_searches") or []
    # take at most one planned search per turn
    if planned:
        s0 = planned[0]
        query = s0.get("search_query","").strip()
        entity_type = s0.get("entity_type","")
        entity_name = s0.get("entity_name","")
        artifact = s0.get("artifact","")
        imgs = google_image_search(query, num=6)
        # validate up to 3 unique
        uniq = []
        seen = set()
        for it in imgs:
            u = (it.get("url") or "").strip()
            if u and u not in seen:
                seen.add(u); uniq.append(it)
            if len(uniq) >= 3: break
        ok_url = ""; reason = ""
        for it in uniq:
            v = agent2_validate_image(
                image_url=it["url"],
                entity_type=entity_type,
                entity_name=entity_name,
                artifact=artifact,
                q_text=last_q,
                a_text=user_text
            )
            if v.get("ok"):
                ok_url = it["url"]; reason = v.get("reason",""); break
        if ok_url:
            sid = next_free_slot(st.session_state.slots)
            if sid:
                label_hint = {
                    "book":"Must-Read", "podcast":"Podcast", "person":"Role Model",
                    "tool":"Go-to Tool", "film":"Influence"
                }.get(entity_type, "Item")
                st.session_state.slots[sid] = {
                    "label": f"{label_hint} — {entity_name}",
                    "image_url": ok_url,
                    "meta": {"type": entity_type, "artifact": artifact, "cluster_id": s0.get("cluster_id","")}
                }

    # Next assistant message (stream)
    slot_summary = ", ".join([s.get("label","") for s in st.session_state.slots.values() if s.get("label")]) or "none yet"
    with st.chat_message("assistant"):
        nxt = agent1_chat_stream(st.session_state.history, st.session_state.mode, slot_summary)
    if not nxt:
        nxt = (plan.get("assistant_message","") or "").strip()
    if nxt:
        st.session_state.history.append({"role":"assistant","content": nxt})
        # Finalize on explicit handoff
        low = nxt.lower()
        if any(p in low for p in HANDOFF_PHRASES):
            data = agent3_finalize(st.session_state.history, st.session_state.slots)
            st.session_state.final_data = data or {}

    st.rerun()

# Final card render + export
if st.session_state.final_data:
    st.subheader("Your Expert Card")
    lines = st.session_state.final_data.get("lines") or []
    html = st.session_state.final_data.get("html") or ""
    order = ["S1","S2","S3","S4"]
    for i, sid in enumerate(order):
        s = st.session_state.slots.get(sid, {})
        title = (s.get("label") or sid).split("—")[-1].strip()
        text = lines[i] if i < len(lines) else ""
        img = (s.get("image_url") or "").strip()
        col_text, col_img = st.columns([3,2], vertical_alignment="center")
        with col_text:
            st.markdown(f"**{title}**")
            st.write(text)
        with col_img:
            if img:
                st.image(img, use_column_width=True)
            else:
                st.caption("(no image)")
    st.download_button("⬇️ Export HTML", data=html.encode("utf-8"), file_name="expert-card.html", mime="text/html")

# Footer actions
cA, cB = st.columns(2)
with cA:
    if st.button("🔄 Restart"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
with cB:
    st.caption("Built with OpenAI Responses API (Structured Outputs, streaming).")
