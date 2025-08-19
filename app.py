import os
import json
import time
from typing import Any, Dict, List, Optional

import streamlit as st
import requests
from openai import OpenAI

# =============================
# Configuration & Clients
# =============================

def get_openai_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.stop()
    return OpenAI(api_key=api_key)

# Optional: Google CSE keys for image search
GOOGLE_CSE_KEY  = st.secrets.get("GOOGLE_CSE_KEY") or os.getenv("GOOGLE_CSE_KEY")
GOOGLE_CSE_CX   = st.secrets.get("GOOGLE_CSE_CX") or os.getenv("GOOGLE_CSE_CX")
GOOGLE_CSE_SAFE = (st.secrets.get("GOOGLE_CSE_SAFE") or os.getenv("GOOGLE_CSE_SAFE") or "active").lower()

# =============================
# Shared Schemas (JSON Schema dicts)
# =============================

STATE_FRAME_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "turn": {"type": "integer"},
        "mode": {"type": "string", "enum": ["professional", "general"]},
        "agenda": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "focus": {"type": "string"},
                "tone": {"type": "string", "enum": ["warm", "neutral", "direct"]},
                "next_move": {"type": "string", "enum": ["clarify", "deepen", "pivot", "close", "handoff"]},
            },
            "required": ["focus", "tone", "next_move"],
        },
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "key": {"type": "string"},
                    "label": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["person", "book", "podcast", "tool", "film", "org", "concept"],
                    },
                    "artifact_preference": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "portrait",
                                "book_cover",
                                "logo",
                                "podcast_cover",
                                "movie_poster",
                            ],
                        },
                    },
                    "best_image_url": {"type": ["string", "null"], "format": "uri"},
                    "source": {"type": ["string", "null"]},
                },
                "required": ["key", "label", "type", "artifact_preference"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": [
                            "authored_by",
                            "hosted_by",
                            "founded_by",
                            "is_logo_of",
                            "related_to",
                        ],
                    },
                },
                "required": ["from", "to", "relation"],
            },
        },
        "search_log": {"type": "array", "items": {"type": "string"}},
        "followups_for_active_item": {"type": "integer"},
        "done_keys": {"type": "array", "items": {"type": "string"}},
        "stop_signal": {"type": "boolean"},
    },
    "required": [
        "turn",
        "mode",
        "agenda",
        "items",
        "relations",
        "search_log",
        "followups_for_active_item",
        "done_keys",
        "stop_signal",
    ],
}

# Orchestrator (Agent 1) requires the model to return an actionable JSON turn output
ORCHESTRATOR_TURN_SCHEMA: Dict[str, Any] = {
    "name": "orchestrator_turn",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "assistant_message": {
                "type": "string",
                "description": "What Agent 1 says to the user this turn (one move).",
            },
            "updated_state": STATE_FRAME_SCHEMA,
            "calls": {
                "type": "array",
                "description": "Optional tool calls to execute this turn.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": ["image_search", "state_commit", "finalize"],
                        },
                        "args": {"type": "object"},
                    },
                    "required": ["name", "args"],
                },
            },
        },
        "required": ["assistant_message", "updated_state"],
    },
    "strict": True,
}

# Finalizer (Agent 3) schema: exactly 4 points + HTML
EXPERT_CARD_SCHEMA: Dict[str, Any] = {
    "name": "expert_card",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "points": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string"},
                        "text": {"type": "string"},
                        "image_url": {"type": ["string", "null"], "format": "uri"},
                    },
                    "required": ["label", "text"],
                },
            },
            "html": {"type": "string"},
        },
        "required": ["points", "html"],
    },
    "strict": True,
}

# =============================
# System Prompts
# =============================

SYSTEM_AGENT1 = """
You are Agent 1 — Interview & Orchestration.

OUTPUT CONTRACT
- Return ONLY JSON in this schema every turn:
  {
    "assistant_message": "string (1–3 sentences to the user)",
    "updated_state": STATE_FRAME,
    "calls": [
      { "name": "image_search", "args": { "query": "...", "artifact_type": "...", "state_snapshot": { ... } } },
      { "name": "state_commit", "args": { "state": STATE_FRAME } }
    ]
  }
- At most ONE image_search call per user turn. If multiple candidate items appear, select the most salient and defer others.
- Always include a final state_commit.

CONTEXT & MODE
- Read the developer-provided state (state.mode ∈ {professional, general}).
- Mirror the user's language (DE/EN/…). Adapt scope & tone:
  • professional → focus on work/decisions/practice.
  • general → include life beyond work (habits, routines, creativity, learning).

ROLE
Lead a short, warm interview to extract 4 strong anchors for an Expert Card. Blend public influences (book, podcast, person, tool, film, org) with the user's lived practices and decisions. Never reveal agents or tools.

CONVERSATION STYLE
- One move per turn: clarify | deepen | pivot | close.
- 1–3 short sentences. Be concrete, specific, no filler.
- If the user asks a question: answer in ≤1 sentence, then continue with your move.

ONTOLOGY
- Searchable public items: person | book | podcast | tool | film | org.
- Non-items: opinions, generic categories, private/internal practices without a public reference.
- Track relations in state.relations: authored_by | hosted_by | founded_by | is_logo_of | related_to.

ARTIFACT MAPPING (for visual slots)
- person→portrait, book→book_cover, podcast→podcast_cover, tool/org→logo, film→movie_poster.
- If both a person and their book appear, prefer ONE visual for the pair (book_cover if the content is the book; otherwise portrait). Record authored_by.

FOLLOW-UPS & FLOW
- Max two deepening turns for the current item; then pivot or close.
- Keep variety across anchor types. If responses are vague or repetitive, pivot.

SEARCH POLICY (model planning)
- At most ONE image search per user turn. Prefer clarify over guessing.
- Dedupe: never schedule a duplicate (key+artifact); maintain state.search_log.

QUERY BUILDING (when planning a search)
- Build precise, artifact-aware queries:
  • book: "<Title> by <Author> book cover" (+ year/publisher if useful)
  • person: "<Full Name> official portrait" | "<Full Name> press photo"
  • podcast: "<Show> podcast cover" | "<Show> by <Host> cover"
  • tool/org: "<Product/Org> official logo"
  • film: "<Title> movie poster" (+ year/director)

CORRECTIONS (natural language)
- Support: drop item, replace item (from→to), prefer variant (e.g., no_text_on_cover, face_only), reassign cluster.
- Reflect corrections in the next turn's state (items/relations/search_log) before proceeding.

STOP CONDITIONS & HANDOFF
- Target 4–6 distinct anchors. After 4 solid items, offer:
  "We have four strong anchors. Would you like me to assemble your 4-point card now, or continue with one or two more?"
- If the user accepts at any time, close with one of:
  "Alright — I’ll now assemble your 4-point card." | "Perfect — I’ll now assemble your selected 4-point card." | "Great — I’ll now assemble your 4-point expert card with your chosen items."
- Hard stop at 6 items → close with a handoff phrase.

TURN LOGIC
1) Read latest user message + state.
2) Choose ONE move (clarify/deepen/pivot/close) and write a short assistant_message.
3) If a search is needed, plan ONE artifact-aware query → emit as image_search call.
4) Always include a state_commit call with the full updated_state.
5) If finalizing, also include a finalize call.
"""

SYSTEM_FINALIZER = """
You are Agent 3 — Finalizer.

GOAL
Produce an Expert Card with exactly 4 points in the user's own voice, grounded in the conversation history and state.items (labels, images, relations).

OUTPUT
Return JSON ONLY per the expert_card schema provided by the caller.

GUIDELINES
- Each point: 1–3 concise sentences, using the user's phrasing when helpful; highlight habits, decisions, before/after, or trade‑offs.
- Avoid generic encyclopedia facts. No tool talk. No instructions.
- Labels should be short and meaningful (e.g., "Must-Read — <Title>", "Role Model — <Name>", "Go‑to Tool — <Tool>", "Influence — <Film>").
- If a slot has no image, set image_url = null. Use relations to avoid redundant person+book visuals.

HTML
- Provide a compact, clean snippet that renders the 4 points (title + text) with optional images.
- If a source/attribution exists in state for an image, include a small source line under the image.
- Keep inline CSS minimal and responsive.
"""

SYSTEM_VALIDATOR_4O = """
You are a vision validator.

Task: Given an expected artifact type and an image URL, decide if the image matches one of:
portrait | book_cover | logo | podcast_cover | movie_poster | screenshot | other.

Rules:
- PERSON policy: Do NOT identify or confirm a specific individual. Only check portrait form criteria (single-human head/shoulders; not group, not cartoon/meme/logo).
- Reject if the image is too small/low‑quality or heavily watermarked/mockup; reject clear mismatches to the requested artifact.

Output format:
- If mismatch or low quality → "reject" (optionally append a very short reason).
- If acceptable → "accept:<one_of_the_types>" followed by a very short reason.
  Examples: "accept:book_cover minimal typography" · "accept:portrait neutral background" · "reject low resolution".
"""

# =============================
# Utilities: Moderation, Image Search, Vision Validation
# =============================

def moderate_text(client: OpenAI, text: str) -> bool:
    try:
        resp = client.moderations.create(input=text)
        flagged = False
        if hasattr(resp, "results"):
            for r in resp.results:
                if getattr(r, "flagged", False):
                    flagged = True
                    break
        return not flagged
    except Exception:
        # On moderation failure, be safe
        return False


def google_image_search(query: str, num: int = 5) -> List[Dict[str, Any]]:
    """Simple Google CSE image search. Returns list of {url, context, mime, width, height}."""
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        return []
    params = {
        "q": query,
        "searchType": "image",
        "num": max(1, min(num, 10)),
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_CX,
        "safe": GOOGLE_CSE_SAFE,
    }
    r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=15)
    if r.status_code != 200:
        return []
    data = r.json()
    out = []
    for item in data.get("items", [])[:num]:
        img = item.get("link")
        ctx = item.get("image", {})
        out.append({
            "url": img,
            "contextLink": item.get("image", {}).get("contextLink") or item.get("link"),
            "mime": ctx.get("mime"),
            "width": ctx.get("width"),
            "height": ctx.get("height"),
        })
    return out


def vision_validate(client: OpenAI, image_url: str, expected: str) -> Dict[str, Any]:
    """Use GPT-4o vision to classify artifact; return {decision, reason}."""
    try:
        res = client.responses.create(
            model="gpt-4o-2024-08-06",
            input=[
                {"role": "system", "content": SYSTEM_VALIDATOR_4O},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"expected={expected}"},
                        {"type": "input_image", "image_url": image_url},
                    ],
                },
            ],
            max_output_tokens=150,
        )
        txt = res.output_text.strip()
    except Exception as e:
        txt = f"error:{e}"
    return {"decision": txt[:100], "raw": txt}


def handle_image_search(client: OpenAI, query: str, artifact_type: str, state_snapshot: Dict[str, Any], use_vision_check: bool = True) -> Dict[str, Any]:
    # Dedupe/cooldown
    if query in state_snapshot.get("search_log", []):
        return {"duplicate": True, "reason": "cooldown", "search_log": state_snapshot.get("search_log", [])}

    candidates = google_image_search(query, num=5)
    if not candidates:
        return {"no_match": True, "reason": "no_search_results"}

    # Heuristic: prefer largest image
    best = sorted(
        candidates,
        key=lambda c: (c.get("width") or 0) * (c.get("height") or 0),
        reverse=True,
    )[0]

    decision = {"decision": "accept:other", "raw": "skipped"}
    if use_vision_check:
        decision = vision_validate(client, best["url"], artifact_type)

    accept = decision.get("decision", "").startswith("accept:")
    if not accept:
        # fallback to first candidate without validation
        pass

    # Build slot update (minimal)
    # key/label/type must be inferred by Agent 1; here we only return the asset
    slot_update = {
        "artifact": artifact_type,
        "best_image_url": best["url"],
        "source": best.get("contextLink"),
    }
    return {
        "slot_update": slot_update,
        "candidates": candidates,
        "duplicate": False,
        "reason": decision.get("decision"),
        "search_log": list({*state_snapshot.get("search_log", []), query}),
    }

# =============================
# Orchestrator Turn
# =============================

def orchestrator_turn(client: OpenAI, user_text: str, state_frame: Dict[str, Any], reasoning_effort: str = "low", verbosity: str = "low") -> Dict[str, Any]:
    resp = client.responses.create(
        model="gpt-5",
        reasoning={"effort": reasoning_effort},        text={"format":{"type":"json_schema","json_schema": ORCHESTRATOR_TURN_SCHEMA, "strict": True}},
        input=[
            {"role": "system", "content": SYSTEM_AGENT1},
            {"role": "developer", "content": json.dumps({"state": state_frame})},
            {"role": "user", "content": user_text},
        ],
        max_output_tokens=800,
        store=True,
    )
    data = resp.output_parsed or {}
    return data

# =============================
# Finalization (Agent 3)
# =============================

def finalize_card(client: OpenAI, history: List[Dict[str, str]], state_frame: Dict[str, Any]) -> Dict[str, Any]:
    resp = client.responses.create(
        model="gpt-5-mini",
        text={"format":{"type":"json_schema","json_schema": EXPERT_CARD_SCHEMA, "strict": True}},
        input=[
            {"role": "system", "content": SYSTEM_FINALIZER},
            {"role": "developer", "content": json.dumps({"state": state_frame, "history": history})},
        ],
        max_output_tokens=1200,
        store=False,
    )
    return resp.output_parsed or {}

# =============================
# Streamlit UI & App State
# =============================

def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of {role, content}
    if "state_frame" not in st.session_state:
        st.session_state.state_frame = {
            "turn": 0,
            "mode": "professional",
            "agenda": {"focus": "", "tone": "warm", "next_move": "clarify"},
            "items": [],
            "relations": [],
            "search_log": [],
            "followups_for_active_item": 0,
            "done_keys": [],
            "stop_signal": False,
        }
    if "card" not in st.session_state:
        st.session_state.card = None


def render_slots(state_frame: Dict[str, Any]):
    st.subheader("Slots / Items")
    cols = st.columns(4)
    for i, item in enumerate(state_frame.get("items", [])[:4]):
        with cols[i % 4]:
            st.markdown(f"**{item.get('label','(unlabeled)')}** · _{item.get('type','?')}_")
            img = item.get("best_image_url")
            if img:
                st.image(img, use_column_width=True)
                if item.get("source"):
                    st.caption(f"Quelle: {item['source']}")


def render_card(card: Dict[str, Any]):
    if not card:
        return
    st.subheader("Final Expert Card")
    st.markdown(card.get("html", ""), unsafe_allow_html=True)
    st.download_button("Download HTML", data=card.get("html", "").encode("utf-8"), file_name="expert_card.html")

# =============================
# Main App
# =============================

def main():
    st.set_page_config(page_title="Expert Card Agents (Single File)", page_icon="🗂️", layout="wide")
    st.title("🗂️ Expert Card – 3-Agent System (Single File)")

    init_state()
    client = get_openai_client()

    with st.sidebar:
        st.header("Settings")
        st.session_state.state_frame["mode"] = st.selectbox("Mode", ["professional", "general"], index=0)
        reasoning_effort = st.selectbox("Reasoning effort (Agent 1)", ["minimal", "low", "medium", "high"], index=1)
        verbosity = st.selectbox("Verbosity (Agent 1)", ["low", "medium", "high"], index=0)
        use_vision = st.toggle("Use 4o vision validation", value=True)
        if st.button("Restart Session"):
            for k in ["history", "state_frame", "card"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # Conversation display
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box
        # Ensure history has an opener from Agent 1 on first run
    if "history" not in st.session_state:
        st.session_state.history = []
    if not st.session_state.history:
        opener = "Was ist ein Buch, Podcast, eine Person, ein Tool oder ein Film, der dich zuletzt stark beeinflusst hat — und wie?"
        st.session_state.history.append({"role": "assistant", "content": opener})

    user_text = st.chat_input("Deine Antwort…")

    # Finalize button
    colA, colB = st.columns([1,1])
    with colA:
        finalize_now = st.button("Finalize now (build card)")
    with colB:
        st.caption("Hinweis: Nach ~4 Items wird automatisch eine Finalisierung vorgeschlagen.")

    if finalize_now and not st.session_state.card:
        with st.spinner("Finalizing…"):
            card = finalize_card(client, st.session_state.history, st.session_state.state_frame)
            # Optional moderation of card HTML as text
            ok = moderate_text(client, json.dumps(card))
            if not ok:
                st.warning("Moderation flagged the output. Please adjust your content and try again.")
            else:
                st.session_state.card = card
                st.success("Card erstellt.")

    # Normal turn handling
    if user_text:
        # Moderation of user input
        if not moderate_text(client, user_text):
            with st.chat_message("assistant"):
                st.warning("Deine Nachricht wurde von der Moderation blockiert.")
        else:
            st.session_state.history.append({"role": "user", "content": user_text})
            with st.spinner("Agent 1 denkt nach…"):
                # Orchestrator call
                data = orchestrator_turn(
                    client,
                    user_text=user_text,
                    state_frame=st.session_state.state_frame,
                    reasoning_effort=reasoning_effort,
                    verbosity=verbosity,
                )

                # Execute any calls requested by the model (in-prompt protocol)
                calls = data.get("calls", []) or []
                for call in calls:
                    name = call.get("name")
                    args = call.get("args", {})
                    if name == "image_search":
                        query = args.get("query") or ""
                        artifact_type = args.get("artifact_type") or "portrait"
                        result = handle_image_search(
                            client,
                            query=query,
                            artifact_type=artifact_type,
                            state_snapshot=st.session_state.state_frame,
                            use_vision_check=use_vision,
                        )
                        # Merge results into the state frame if slot_update present
                        slot_update = result.get("slot_update")
                        if slot_update:
                            # Heuristic: attach to the last mentioned item if any
                            # In production you'd map by key returned by Agent 1; here we merge by last item
                            if st.session_state.state_frame["items"]:
                                i = -1
                                st.session_state.state_frame["items"][i]["best_image_url"] = slot_update.get("best_image_url")
                                st.session_state.state_frame["items"][i]["source"] = slot_update.get("source")
                            # Update search log
                            if result.get("search_log"):
                                st.session_state.state_frame["search_log"] = list(result["search_log"])  # idempotent

                    elif name == "state_commit":
                        # The model can propose a full updated state; we'll accept if it validates basic keys
                        proposed = args.get("state") or {}
                        # Minimal guard: ensure required keys exist
                        for k in [
                            "turn",
                            "mode",
                            "agenda",
                            "items",
                            "relations",
                            "search_log",
                            "followups_for_active_item",
                            "done_keys",
                            "stop_signal",
                        ]:
                            if k not in proposed:
                                break
                        else:
                            st.session_state.state_frame = proposed

                    elif name == "finalize":
                        # Trigger finalization immediately
                        st.session_state.card = finalize_card(
                            client, st.session_state.history, st.session_state.state_frame
                        )

                # Assistant message to user
                assistant_message = data.get("assistant_message", "")
                if assistant_message:
                    st.session_state.history.append({"role": "assistant", "content": assistant_message})

                # Update state from model (authoritative)
                updated_state = data.get("updated_state")
                if updated_state:
                    st.session_state.state_frame = updated_state

            # Render last assistant message in UI
            with st.chat_message("assistant"):
                st.markdown(st.session_state.history[-1]["content"] if st.session_state.history else "")

    # Right column panels
    with st.expander("State (debug)"):
        st.json(st.session_state.state_frame)

    render_slots(st.session_state.state_frame)
    if st.session_state.card:
        render_card(st.session_state.card)


if __name__ == "__main__":
    main()
