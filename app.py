# app.py
# 🟡 Expert Card — 3-Agent System (Single File, Streamlit + OpenAI Responses API)
# - Start-Gate (kein Model-Call vor Klick)
# - Agent 1 (Interview & Orchestration) => Structured Outputs (json_schema via Responses.text.format)
# - Agent 2 (Image-Validator) => Structured Outputs (json_schema)
# - Agent 3 (Finalizer) => Structured Outputs (json_schema)
# - Google CSE Bildersuche (1 Suche pro Turn), strikte Personen-Policy (keine Identifizierung)
# - Slots + HTML-Export

import os
import json
import time
import uuid
import requests
import streamlit as st
from typing import Any, Dict, List, Optional
from openai import OpenAI

# ----------------------------
# Config & Keys
# ----------------------------
APP_TITLE = "🟡 Expert Card — 3 Agents (Streamlit · OpenAI Responses)"
MODEL = os.getenv("OPENAI_GPT5_SNAPSHOT", "gpt-5")
AGENT2_MODEL = os.getenv("OPENAI_AGENT2_MODEL", "gpt-5-mini")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY", "").strip()
GOOGLE_CSE_CX  = os.getenv("GOOGLE_CSE_CX", "").strip()
GOOGLE_CSE_SAFE = os.getenv("GOOGLE_CSE_SAFE", "off").strip().lower()  # "off"|"active"

if not OPENAI_API_KEY:
    st.stop()

# ----------------------------
# OpenAI Client
# ----------------------------
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# JSON Schemas (Structured Outputs)
# ----------------------------
ORCHESTRATOR_TURN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "assistant_message": {"type": "string"},
        "updated_state": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["professional", "general"]},
                "items": {"type": "object", "additionalProperties": True},
                "relations": {"type": "object", "additionalProperties": True},
                "search_log": {"type": "array", "items": {"type": "string"}},
                "flow": {"type": "object", "additionalProperties": True},
                "followup_count": {"type": "object", "additionalProperties": True},
                "stop_signal": {"type": "boolean"}
            },
            "additionalProperties": True
        },
        "calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [
                            "image_search", "state_commit", "finalize",
                            "drop_item", "replace_item", "reassign_cluster", "prefer_variant"
                        ]
                    },
                    "args": {"type": "object", "additionalProperties": True}
                },
                "required": ["name", "args"],
                "additionalProperties": False
            }
        }
    },
    "required": ["assistant_message", "updated_state"],
    "additionalProperties": False
}

VALIDATOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "reason": {"type": "string"}
    },
    "required": ["ok"],
    "additionalProperties": False
}

EXPERT_CARD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "text": {"type": "string"},
                    "image_url": {"type": ["string", "null"]},
                    "source_url": {"type": ["string", "null"]}
                },
                "required": ["label", "text", "image_url"],
                "additionalProperties": False
            },
            "minItems": 4,
            "maxItems": 4
        },
        "html": {"type": "string"}
    },
    "required": ["points", "html"],
    "additionalProperties": False
}

# ----------------------------
# Prompts (System)
# ----------------------------
SYSTEM_AGENT1 = """You are Agent 1 — Interview & Orchestration.

OUTPUT CONTRACT
Return ONLY JSON according to the provided schema (assistant_message, updated_state, calls).
- At most ONE image_search call per user turn. If multiple candidate items appear, select the most salient and defer others.
- Always include a state_commit call.

CONTEXT & MODE
- Use state.mode ∈ {professional, general}. Mirror user language (DE/EN).
- professional → focus on work/decisions/practice.
- general → include life beyond work (habits, routines, creativity).

ROLE
Lead a short, warm interview to extract 4 strong anchors for an Expert Card.
Blend public items (person, book, podcast, tool, film, org) with the user's practices.
Never reveal tools/agents.

STYLE
- One move/turn: clarify | deepen | pivot | close.
- 1–3 short sentences, concrete, no filler.
- If the user asks you something, answer in ≤1 sentence, then proceed.

ONTOLOGY
- Searchable public items: person | book | podcast | tool | film | org.
- Track relations: authored_by | hosted_by | founded_by | is_logo_of | related_to.

ARTIFACT MAP
- person→portrait, book→book_cover, podcast→podcast_cover, tool/org→logo, film→movie_poster.
- If person+book both appear, prefer ONE visual (book_cover if focus is the book; otherwise portrait). Record authored_by.

FLOW
- Max two deepening turns per current item; then pivot or close.
- Keep variety across types. If vague/repetitive, pivot.

SEARCH POLICY
- Plan at most ONE search per user turn. Deduplicate by (key+artifact) using state.search_log.

QUERY BUILDER
- book: "<Title> by <Author> book cover" (+year/publisher if available)
- person: "<Full Name> official portrait" or "press photo"
- podcast: "<Show> podcast cover" or "<Show> by <Host> cover"
- tool/org: "<Product/Org> official logo"
- film: "<Title> movie poster" (+year/director)

CORRECTIONS
Support: drop item, replace item (from→to), prefer variant (no_text_on_cover|face_only), reassign cluster.
Update state accordingly before continuing.

STOP & HANDOFF
- Target 4–6 anchors. After 4, offer to finalize. On acceptance, end with a handoff phrase.
- If hard-stop triggered in state.stop_signal, close now with a handoff message.

TURN LOGIC
1) Read latest user_text + state.
2) Choose ONE move (clarify/deepen/pivot/close) and write assistant_message.
3) If search is needed, emit ONE image_search in calls; otherwise none.
4) Always include a state_commit (with full updated_state). If finalizing, also include finalize call.
"""

SYSTEM_VALIDATOR = """You are an image validator. Return ONLY JSON per the provided schema.
Policy for PERSON portraits: Do NOT identify or confirm specific individuals.
Check ONLY portrait form criteria: single human, proper head/shoulders framing, not group, not logo/cartoon/meme.

General checks:
- If artifact=book_cover: plausible book cover (accept variants; reject plain portraits/logos).
- podcast_cover: square-ish artwork with show/host branding (reject generic unrelated stock).
- logo: official product/org logo (reject unrelated brand).
- movie_poster: poster-like composition (title/credits preferred; reject random stills).

Return {"ok": true/false, "reason": "..."}.
"""

SYSTEM_FINALIZER = """You are Agent 3 — Finalizer. Return ONLY JSON matching the provided schema.

Goal
Produce exactly 4 points in the user's own voice, grounded in the conversation and state.items (labels, images, relations).

Guidelines
- Each point: 1–3 concise sentences; highlight habits, decisions, before/after, or trade-offs.
- No encyclopedic fluff. No tool talk. No instructions.
- Labels short: "Must-Read — <Title>", "Role Model — <Name>", "Go-to Tool — <Tool>", "Influence — <Film>".
- If a slot has no image, image_url = null. Include a small source_url if available for attribution inside JSON.
- Also produce minimal, responsive HTML snippet that renders the 4 points (title + text + optional image + optional source link).
"""

# ----------------------------
# Helpers
# ----------------------------
def _resp_text(resp) -> str:
    t = getattr(resp, "output_text", None)
    if t:
        return t
    try:
        # fallback: join message outputs (Responses API)
        outs = getattr(resp, "output", []) or []
        parts = []
        for it in outs:
            if getattr(it, "type", "") == "message":
                content = getattr(it, "content", []) or []
                for c in content:
                    if getattr(c, "type", "") in ("output_text", "text"):
                        parts.append(getattr(c, "text", "") or "")
        return "".join(parts)
    except Exception:
        return ""

def safe_json(obj_text: str) -> Dict[str, Any]:
    try:
        return json.loads(obj_text)
    except Exception:
        return {}

def normalize_key(entity_type: str, entity_name: str) -> str:
    et = (entity_type or "").strip().lower()
    nm = (entity_name or "").strip().lower()
    return f"{et}|{nm}"

# ----------------------------
# Google CSE Image Search
# ----------------------------
def google_image_search(query: str, num: int = 6) -> List[Dict[str, str]]:
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        return []
    params = {
        "q": query,
        "searchType": "image",
        "num": max(1, min(10, num)),
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
            ctx = ""
            imgobj = it.get("image") or {}
            ctx = (imgobj.get("contextLink") or "").strip()
            if link:
                out.append({"url": link, "page_url": ctx})
        return out
    except Exception:
        return []

# ----------------------------
# Agent 2: Validator (Structured Outputs via json_schema)
# ----------------------------
def validate_image_with_context_for_artifact(
    client: OpenAI,
    image_url: str,
    entity_type: str,
    entity_name: str,
    artifact: str,
    q_text: str,
    a_text: str,
) -> Dict[str, Any]:
    # Build input text w/o identity confirmation (person policy)
    if (entity_type or "").strip().lower() == "person" and (artifact or "").strip().lower() == "portrait":
        user_text = (
            "Artifact: portrait\n"
            "Item type: person\n"
            "Person name (context only, do NOT identify): " + (entity_name or "") + "\n"
            "Check only portrait form criteria. Return JSON."
        )
    else:
        user_text = (
            f"Artifact: {artifact}\n"
            f"Item type: {entity_type}\n"
            f"Item name: {entity_name}\n"
            f"Context Q: {q_text[:300]}\n"
            f"Context A: {a_text[:500]}\n"
            "Does this image plausibly match the requested public item? Return JSON."
        )
    resp = client.responses.create(
        model=AGENT2_MODEL,
        text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "validator",
                "schema": VALIDATOR_SCHEMA,
                "strict": True,
            }
        },
        input=[
            {"role": "system", "content": SYSTEM_VALIDATOR},
            {"role": "user", "content": [
                {"type": "input_text", "text": user_text},
                {"type": "input_image", "image_url": image_url}
            ]}
        ]
    )
    data = safe_json(_resp_text(resp))
    ok = bool(data.get("ok"))
    reason = (data.get("reason") or "")[:200]
    return {"ok": ok, "reason": reason}

# ----------------------------
# Agent 3: Finalizer
# ----------------------------
def agent3_finalize(client: OpenAI, history: List[Dict[str, str]], state: Dict[str, Any]) -> Dict[str, Any]:
    # Prepare compact transcript (last 24 turns)
    convo = []
    for m in history[-24:]:
        prefix = "Q: " if m["role"] == "assistant" else "A: "
        convo.append(prefix + m["content"])
    transcript = "\n".join(convo)

    resp = client.responses.create(
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
        input=[
            {"role": "system", "content": SYSTEM_FINALIZER},
            {"role": "user", "content": json.dumps({
                "transcript": transcript,
                "state": state
            }, ensure_ascii=False)}
        ]
    )
    data = safe_json(_resp_text(resp))
    return data

# ----------------------------
# Agent 1: Orchestrator Turn
# ----------------------------
def orchestrator_turn(client: OpenAI, user_text: str, state_frame: Dict[str, Any]) -> Dict[str, Any]:
    resp = client.responses.create(
        model=MODEL,
        text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "OrchestratorTurn",
                "schema": ORCHESTRATOR_TURN_SCHEMA,
                "strict": True
            }
        },
        input=[
            {"role": "system", "content": SYSTEM_AGENT1},
            {"role": "user", "content": json.dumps({
                "user_text": user_text,
                "state": state_frame
            }, ensure_ascii=False)}
        ]
    )
    txt = _resp_text(resp)
    return safe_json(txt)

# ----------------------------
# Session State
# ----------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("user_started", False)
    ss.setdefault("agent1_mode", "professional")  # "professional"|"general"
    ss.setdefault("history", [])
    ss.setdefault("final_card", None)

    # Orchestrator state frame
    ss.setdefault("state_frame", {
        "mode": "professional",
        "items": {},            # slot_id -> { key, label, item_type, artifact, image_url, source_url }
        "relations": {},
        "search_log": [],
        "flow": {"current_key": None, "done_keys": []},
        "followup_count": {},
        "stop_signal": False
    })

# ----------------------------
# Slot Helpers
# ----------------------------
def next_free_slot(items: Dict[str, Any]) -> Optional[str]:
    for sid in ["S1", "S2", "S3", "S4"]:
        if sid not in items:
            return sid
    return None

def upsert_slot_dedupe(
    items: Dict[str, Any], key: str, artifact: str, entity_name: str,
    image_url: str, source_url: str, label_hint: Optional[str] = None, cluster_id: Optional[str] = None
) -> str:
    # dedupe by same key+artifact
    for sid, s in items.items():
        if s.get("key") == key and s.get("artifact") == artifact:
            s["label"] = s.get("label") or (label_hint or "Item") + " — " + entity_name
            s["image_url"] = image_url
            if source_url:
                s["source_url"] = source_url
            if cluster_id:
                s["cluster"] = cluster_id
            items[sid] = s
            return sid
    sid = next_free_slot(items) or f"X{len(items)+1}"
    items[sid] = {
        "key": key,
        "label": (label_hint or "Item") + " — " + entity_name,
        "item_type": key.split("|", 1)[0],
        "artifact": artifact,
        "image_url": image_url,
        "source_url": source_url or None,
        "cluster": cluster_id or ""
    }
    return sid

# ----------------------------
# UI Helpers
# ----------------------------
def render_slots(items: Dict[str, Any]):
    st.subheader("Slots")
    cols = st.columns(2)
    order = ["S1","S2","S3","S4"]
    for idx, sid in enumerate(order):
        with cols[idx % 2]:
            s = items.get(sid)
            if not s:
                st.info(f"{sid}: (leer)")
                continue
            st.markdown(f"**{sid} — {(s.get('label') or '').split('—')[-1].strip()}**")
            img = s.get("image_url") or ""
            if img:
                st.markdown(
                    f'<img src="{img}" style="max-width:100%;border-radius:12px;border:1px solid rgba(0,0,0,.06);" />',
                    unsafe_allow_html=True
                )
                if s.get("source_url"):
                    st.caption(f"Quelle: {s['source_url']}")
            else:
                st.caption("(kein Bild)")

def render_history():
    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def render_progress(items: Dict[str, Any]):
    filled = sum(1 for s in items.values() if (s.get("label") or "").strip())
    st.progress(min(1.0, filled/4.0), text=f"Fortschritt: {filled}/4")

def render_final_card(card: Dict[str, Any]):
    if not card:
        return
    st.subheader("Your Expert Card")
    pts = card.get("points", []) or []
    for p in pts:
        with st.container():
            st.markdown(f"**{p.get('label','')}**")
            st.write(p.get("text",""))
            img = p.get("image_url")
            if img:
                st.markdown(
                    f'<img src="{img}" style="max-width:100%;border-radius:12px;border:1px solid rgba(0,0,0,.06);" />',
                    unsafe_allow_html=True
                )
            if p.get("source_url"):
                st.caption(f"Quelle: {p['source_url']}")
    html = card.get("html","")
    if html:
        st.download_button("⬇️ Export HTML", data=html.encode("utf-8"), file_name="expert-card.html", mime="text/html")

# ----------------------------
# MAIN
# ----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
init_state()

# Start-Gate (keine Model-Aufrufe vor Klick)
st.info("Klicke **Start**, um das Interview zu beginnen. Bis dahin werden **keine** Modell-Aufrufe ausgeführt.")
c1, c2 = st.columns([1,4])
with c1:
    if st.button("▶️ Start"):
        st.session_state.user_started = True
        st.session_state.history = []
        st.session_state.final_card = None
        st.rerun()

if not st.session_state.user_started:
    st.stop()

# Mode selection (beeinflusst Agent-1 Verhalten)
st.session_state.agent1_mode = st.radio(
    "Interview focus",
    options=["professional", "general"],
    index=0 if st.session_state.agent1_mode == "professional" else 1,
    horizontal=True
)
st.session_state.state_frame["mode"] = st.session_state.agent1_mode

# Opener: Nur nach Start, ohne vorherige Model-Aufrufe
if not st.session_state.history:
    # Erster Turn: Agent 1 erzeugt eine Eröffnungsfrage (keine Suche im ersten Turn)
    client = get_openai_client()
    data = orchestrator_turn(client, user_text="", state_frame=st.session_state.state_frame)
    opener = (data.get("assistant_message") or "").strip()
    if opener:
        st.session_state.history.append({"role": "assistant", "content": opener})
    # Commit initial state falls vorhanden
    if "updated_state" in data:
        st.session_state.state_frame = data["updated_state"]

# Live UI
render_progress(st.session_state.state_frame["items"])
render_slots(st.session_state.state_frame["items"])
render_history()

# Final Card anzeigen, falls vorhanden
render_final_card(st.session_state.final_card)

# Chat Input
user_text = st.chat_input("Dein Zug …")
if user_text:
    st.session_state.history.append({"role": "user", "content": user_text})
    client = get_openai_client()

    # Orchestrator Turn
    with st.spinner("Agent 1 denkt nach…"):
        data = orchestrator_turn(client, user_text=user_text, state_frame=st.session_state.state_frame)

    # Assistant message
    msg = (data.get("assistant_message") or "").strip()
    if msg:
        st.session_state.history.append({"role": "assistant", "content": msg})

    # Update state (commit first)
    if "updated_state" in data:
        st.session_state.state_frame = data["updated_state"]

    # Execute calls (max one image_search per turn enforced by Agent 1; we still guard)
    calls = data.get("calls", []) or []
    did_search = False
    for call in calls:
        name = call.get("name")
        args = call.get("args", {}) or {}

        if name == "image_search" and not did_search:
            did_search = True
            query = args.get("query") or ""
            artifact = (args.get("artifact_type") or "portrait").strip().lower()
            entity_type = (args.get("entity_type") or "").strip().lower()
            entity_name = (args.get("entity_name") or "").strip()
            cluster_id = (args.get("cluster_id") or "").strip()
            label_hint = {
                "book": "Must-Read",
                "podcast": "Podcast",
                "person": "Role Model",
                "tool": "Go-to Tool",
                "film": "Influence",
                "org": "Organization"
            }.get(entity_type, "Item")

            # Dedupe by key+artifact
            key = normalize_key(entity_type, entity_name)
            dedupe_token = f"{key}|{artifact}"
            if dedupe_token in st.session_state.state_frame["search_log"]:
                pass
            else:
                st.session_state.state_frame["search_log"].append(dedupe_token)
                # Perform Google Image Search
                results = google_image_search(query, num=6)
                # Validate up to 3 unique candidates
                checked = 0
                picked = None
                seen = set()
                last_assistant_q = ""
                # find last assistant question for context
                for m in reversed(st.session_state.history[:-1]):
                    if m["role"] == "assistant":
                        last_assistant_q = m["content"]
                        break

                for it in results:
                    url = (it.get("url") or it.get("link") or "").strip()
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    checked += 1
                    v = validate_image_with_context_for_artifact(
                        client, url, entity_type, entity_name, artifact,
                        q_text=last_assistant_q, a_text=user_text
                    )
                    if v.get("ok"):
                        picked = {"url": url, "source": it.get("page_url", ""), "reason": v.get("reason","")}
                        break
                    if checked >= 3:
                        break

                if picked:
                    sid = upsert_slot_dedupe(
                        st.session_state.state_frame["items"],
                        key=key, artifact=artifact, entity_name=entity_name,
                        image_url=picked["url"], source_url=picked.get("source",""),
                        label_hint=label_hint, cluster_id=cluster_id or ""
                    )

        elif name == "state_commit":
            # Already applied updated_state above; nothing else to do.
            pass

        elif name == "finalize":
            # Trigger Finalizer
            with st.spinner("Finalisiere Card …"):
                card = agent3_finalize(client, st.session_state.history, st.session_state.state_frame)
                st.session_state.final_card = card

        elif name == "drop_item":
            # Optional maintenance
            skey = (args.get("slot_key") or "").strip()
            if skey and skey in st.session_state.state_frame["items"]:
                del st.session_state.state_frame["items"][skey]

        elif name == "replace_item":
            # Simplistic replace => schedule as new search next turn (Agent 1 should plan it)
            pass

        elif name == "reassign_cluster":
            # Update cluster on given slot
            skey = (args.get("slot_key") or "").strip()
            newc = (args.get("cluster_id") or "").strip()
            if skey and newc and skey in st.session_state.state_frame["items"]:
                st.session_state.state_frame["items"][skey]["cluster"] = newc

        elif name == "prefer_variant":
            # Variant handled by Agent 1 as another image_search in a later turn
            pass

    st.rerun()
