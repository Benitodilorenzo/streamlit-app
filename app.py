import os
import re
import json
import uuid
import urllib.parse
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI

# =====================================
# STREAMLIT OPTIONS (avoid inotify issues in web UIs)
# =====================================
# Use polling-based file watcher to prevent inotify EMFILE errors in hosted environments
st.set_option("server.fileWatcherType", "poll")
st.set_option("server.folderWatchBlacklist", [
    "/.git/", "/.venv/", "/venv/", "/node_modules/", "/__pycache__/", "/.mypy_cache/"
])

# =====================================
# CONFIGURATION
# =====================================
APP_TITLE = "🟡 Expert Card Creator — Two-Agent Architecture"

# Environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5")             # main interview/orchestration agent
MEDIA_MODEL = os.getenv("OPENAI_MEDIA_MODEL", CHAT_MODEL)        # media agent (can be same)
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")     # image generation

# Limits & Defaults
GLOBAL_QUESTION_CAP = 12
MAX_SLOTS = 4
MAX_CANDIDATES = 3

# UI flags
SHOW_DEBUG = False

# =====================================
# SYSTEM PROMPTS (AGENTS)
# =====================================
CHAT_AGENT_SYSTEM_PROMPT = """
You are the single Interview & Orchestration Agent for creating a 4-point “Expert Card”.

[PRIMARY OBJECTIVE]
Deliver a crisp 4-point Expert Card about the user:
- Exactly 4 labeled items (labels are flexible, not hard-coded to “Book/Podcast/…”).
- Each item: 1–2 sentence final line, grounded in the user’s answers.
- Keep a subtle overarching lens: data & AI strategy/design when relevant; do not force it.

[STYLE]
Warm, professional, concrete. No flattery, no filler.
Ask EXACTLY ONE short question per turn. Avoid confirmations and double-questions.

[INTERVIEW FLOW / SLOT LOGIC]
- Maintain up to 4 slots. For each slot:
  1) Elicit specifics (what/why/impact/example) with at most 1–2 follow-ups.
  2) When sufficient signal is present, pick a precise label (e.g., “Role Model”, “Must-Read”, “Go-to Tool”, “Big Idea”, “Design Principle”, “Practical Habit”, “Contrarian View”).
  3) Store 2–4 short internal bullets (facts/evidence/phrasing cues).
  4) If the item is public (e.g., person, tool, book, podcast), request media with MEDIA_REQUEST (intent + focused query string).
  5) If purely conceptual/private or no public item exists, request generation (MEDIA_REQUEST with intent="generic").

- Move to the next slot when:
  (a) you have a clear label + 2–4 bullets, or
  (b) the user indicates no interest/knowledge here.

- Stop after 4 slots or if the conversation is exhausted.

[MEDIA INTEGRATION]
- After MEDIA_RESULT for a slot:
  - If status=found|generated and best_image_url is non-empty, mark slot.media.status accordingly.
  - If status=none, continue the interview; you may try one alternate MEDIA_REQUEST only if new signal appears.
- Do not reveal internal media decisions; keep the chat focused on the interview.

[GUARDRAILS]
- Ask one question per turn. Never list multiple questions.
- Keep the card grounded in the user’s content. No invention of facts, affiliations, or identities.
- Only hint at data & AI if the user’s content permits. If not, choose labels that fit (e.g., “Creative Fuel”, “Learning Habit”).
- Avoid prescribing tools/frameworks the user did not mention.
- Be robust to outliers: non-answers, counter-questions, jokes. Gently steer back with a single, concrete question.
- Never request or rely on sensitive personal data; decline gracefully if offered.

[THEMENSPEKTRUM (EXAMPLES, NOT MANDATORY)]
- Role Model / Influential Thinker
- Must-Read / High-Impact Article/Book
- Go-to Tool / Framework / Method
- Big Idea / Contrarian View
- Practice / Ritual (e.g., “Whiteboard first”)
- Favorite Podcast / Channel
- Signature Use Case
- Design Principle (e.g., “Data before Models”)
Pick labels that match what the user actually gives you.

[AUSREIẞER & UNVORHERGESEHENES]
- If the user answers with jokes or metaphors only: ask one clarifying, concrete question to anchor to a real item or practice.
- If the user asks you a question: answer briefly (1–2 sentences) and then ask exactly one next interview question.
- If the user refuses a topic: acknowledge once, switch to a different angle (e.g., “practice” instead of “book”).
- If the user mentions a private or non-public item: skip media search and continue the slot textually.

[ONE-QUESTION TACTICS]
- Prefer narrow prompts: “Which single book shaped how you approach data strategy? Title only.”
- Follow-ups focus on impact: “What changed in your practice after this?” or “Where did it save time/money/risk?”

[FINALIZATION]
When 4 slots are ready (labels + bullets), produce the final card by calling FINALIZE_CARD(notes=[{label, bullets}]).
Then thank the user succinctly.

[INITIAL GREETING]
Start immediately with a single, concrete question to open slot 1.
"""

MEDIA_AGENT_SYSTEM_PROMPT = """
You are the Media Agent for the Expert Card. You receive MEDIA_REQUESTs and return MEDIA_RESULTs.
Your job: find 2–3 strong image candidates and pick ONE best validated URL; if none are suitable, generate a tasteful fallback.

[INPUT]
MEDIA_REQUEST { slot_id, intent in {person|book|podcast|tool|generic}, query, constraints }

[OUTPUT]
MEDIA_RESULT { slot_id, status in {found|generated|none}, best_image_url, candidates[], notes }

[SEARCH STRATEGY]
- At most 3 focused lookups per request.
- Prioritize authoritative/official sources:
  * Person → wikipedia.org (lead image), official site, reputable media.
  * Book → publisher site, Amazon, OpenLibrary, Wikipedia cover (avoid fan art).
  * Podcast → Apple/Spotify/YouTube official channel art.
  * Tool → official website/docs/press kit (avoid scraped logo aggregators).
- Reject low-resolution, watermarked, or mislabelled images.

[VALIDATION RULES]
- best_image_url MUST be http(s) and directly renderable.
- Reject if the image mismatches the query (wrong person/book/tool).
- Prefer canonical images over memes.
- Return distinct top 2–3 in candidates.

[GENERATION FALLBACK]
- If intent="generic" or search fails → generate a minimal, tasteful graphic:
  * No real faces. No brand logos unless explicitly provided.
  * Style: clean, flat, high-contrast; transparent background preferred.
  * Prompt built ONLY from provided query.
- Return status="generated" and the generated URL.

[COMPLIANCE]
- Do not synthesize public figures’ faces or copyrighted artwork.
- Include page_url for attribution when available.

[ROBUSTNESS]
- If the query is vague, refine internally (append “official site” or “publisher cover”), but never exceed 3 lookups.
- If none pass validation, return status="none" with notes.

[FORMAT]
- candidates: up to 3 objects {image_url, page_url, title}
- best_image_url: a single chosen URL or empty string.
- notes: brief selection/validation rationale.
"""

FINALIZER_SYSTEM_PROMPT = """
You turn structured notes into a concise, upbeat Expert Card.
Return 4 labeled items with 1–2 sentences each, specific and grounded in the notes.
No fluff, no invention, professional and warm.
Format:
- Label: line
- Label: line
- Label: line
- Label: line
"""

# Used by media agent to coax LLM to collect candidates (when no live web tool is available)
SEARCHER_SYSTEM = """
You are a researcher using a web search tool.
Find high-quality candidate images that best represent the requested item.
Heuristics:
- Book: publisher/Amazon/Open Library/Wikipedia; avoid fan art.
- Podcast: official channel art (YouTube/Spotify/Apple) or website.
- Person: clear, respectful portrait (Wikipedia/official site preferred).
- Tool: official logo/hero image.
IMPORTANT: Perform at most 3 focused searches. Stop early if you have 2–3 strong candidates.
Output 1–3 lines. Each line EXACTLY:
IMAGE: <direct_image_url> | SOURCE: <source_page_url>
No extra commentary.
"""

VISION_PICKER_PROMPT = """
You are a careful visual validator.
Pick the SINGLE best candidate URL from the list that matches the item (book cover / podcast art / person portrait / tool logo).
Prefer official or well-known sources, clarity, and correctness.
Output EXACTLY the chosen IMAGE url (no comments). If none suitable, output: NONE
"""

# =====================================
# OpenAI Client
# =====================================

def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it in Streamlit → Settings → Secrets.")
    return OpenAI(api_key=OPENAI_API_KEY)

client: Optional[OpenAI] = None  # will be set in main

# =====================================
# Tools (Function-Calling) for Chat Agent
# =====================================
CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "MEDIA_REQUEST",
            "description": "Ask the Media Agent to find or generate an image for a slot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slot_id": {"type": "string"},
                    "intent": {"type": "string", "enum": ["person", "book", "podcast", "tool", "generic"]},
                    "query": {"type": "string"},
                    "constraints": {
                        "type": "object",
                        "properties": {
                            "max_candidates": {"type": "integer", "default": MAX_CANDIDATES},
                            "must_be_official": {"type": "boolean", "default": True},
                            "prefer_domains": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "required": ["slot_id", "intent", "query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "FINALIZE_CARD",
            "description": "Create the final 4 labeled lines (1–2 sentences each) from collected notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "bullets": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["label", "bullets"]
                        }
                    }
                },
                "required": ["notes"]
            }
        }
    }
]

# =====================================
# Utilities
# =====================================

def is_http_url(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s2 = s.strip()
    if not (s2.startswith("http://") or s2.startswith("https://")):
        return False
    try:
        p = urllib.parse.urlparse(s2)
        return bool(p.scheme and p.netloc)
    except Exception:
        return False


def enforce_single_question(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"[\n\r]|[?]|(?<=\.)\s", text.strip(), maxsplit=1)
    out = parts[0].strip()
    if "?" in text and not out.endswith("?"):
        out += "?"
    return out


def chat_once(model: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None,
              tool_choice: Optional[str] = None) -> Any:
    kwargs: Dict[str, Any] = {"model": model, "messages": messages}
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs)

# =====================================
# Media Agent (host-implemented functions)
# =====================================

def media_search_candidates(query: str) -> List[Dict[str, str]]:
    """Use LLM-guided search (no external web APIs here) to output up to 3 formatted candidates."""
    txt = chat_once(MEDIA_MODEL, [
        {"role": "system", "content": SEARCHER_SYSTEM},
        {"role": "user", "content": f"Query: {query}"}
    ]).choices[0].message.content or ""

    out: List[Dict[str, str]] = []
    for ln in txt.splitlines():
        s = ln.strip()
        if not s or not s.upper().startswith("IMAGE:"):
            continue
        payload = s.split(":", 1)[1].strip() if ":" in s else s
        parts = [p.strip() for p in payload.split("|")]
        image_url = parts[0]
        page_url = ""
        if len(parts) > 1 and parts[1].lower().startswith("source:"):
            page_url = parts[1].split(":", 1)[1].strip()
        if is_http_url(image_url):
            out.append({"image_url": image_url, "page_url": page_url, "title": query})
        if len(out) >= MAX_CANDIDATES:
            break
    return out


def media_pick_best(item_desc: str, candidates: List[Dict[str, str]]) -> str:
    if not candidates:
        return ""
    cand_lines = [f"IMAGE: {c['image_url']} | SOURCE: {c.get('page_url','')}" for c in candidates if is_http_url(c["image_url"])]
    if not cand_lines:
        return ""
    txt = chat_once(MEDIA_MODEL, [
        {"role": "system", "content": VISION_PICKER_PROMPT},
        {"role": "user", "content": f"ITEM:\n{item_desc}\n\nCANDIDATES:\n" + "\n".join(cand_lines)}
    ]).choices[0].message.content or ""
    chosen = txt.strip()
    if chosen.upper() == "NONE":
        return ""
    m = re.search(r"https?://\S+", chosen)
    return m.group(0) if m and is_http_url(m.group(0)) else ""


def media_generate_image(prompt: str) -> str:
    try:
        r = client.images.generate(model=IMAGE_MODEL, prompt=prompt, size="1024x1024", n=1)
        url = r.data[0].url
        return url if is_http_url(url) else ""
    except Exception:
        return ""


def media_agent_handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Implements the Media Agent policy synchronously in host code."""
    slot_id = payload.get("slot_id", "")
    intent = payload.get("intent", "generic")
    query = payload.get("query", "").strip()

    result: Dict[str, Any] = {
        "type": "MEDIA_RESULT",
        "slot_id": slot_id,
        "status": "none",
        "best_image_url": "",
        "candidates": [],
        "notes": ""
    }

    if not query:
        result["notes"] = "Empty query."
        return result

    candidates = media_search_candidates(query)
    result["candidates"] = candidates

    if candidates:
        item_desc = f"{intent.title()}: {query}"
        best = media_pick_best(item_desc, candidates)
        if is_http_url(best):
            result["status"] = "found"
            result["best_image_url"] = best
            result["notes"] = "Picked validated candidate."
            return result

    if intent == "generic" or not candidates:
        gen_prompt = (
            "Create a clean, stylized icon representing: "
            + query
            + ". No text, no real faces, no brand logos. Flat style, high contrast, transparent background."
        )
        gen_url = media_generate_image(gen_prompt)
        if is_http_url(gen_url):
            result["status"] = "generated"
            result["best_image_url"] = gen_url
            result["notes"] = "Generated fallback icon."
            return result

    result["notes"] = "No valid candidate and generation not applied."
    return result

# =====================================
# Finalization (host-side)
# =====================================

def finalize_card_from_notes(notes: List[Dict[str, Any]]) -> Dict[str, str]:
    content = "Create the final Expert Card with 4 labeled items:\n" + "\n".join(
        f"- {n['label']}: " + "; ".join(n.get("bullets", [])) for n in notes
    )
    r = chat_once(CHAT_MODEL, [
        {"role": "system", "content": FINALIZER_SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ])
    txt = r.choices[0].message.content or ""
    lines = [l.strip("- ").strip() for l in txt.splitlines() if l.strip()]
    out: Dict[str, str] = {}
    for l in lines:
        if ":" in l:
            label, body = l.split(":", 1)
            out[label.strip()] = body.strip()
    return out

# =====================================
# Streamlit State
# =====================================

def init_state():
    if "history" not in st.session_state:
        st.session_state.history = [
            {"role": "system", "content": CHAT_AGENT_SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi! Which single book most changed how you think about building data-driven products? Title only."}
        ]
    if "slots" not in st.session_state:
        st.session_state.slots = []  # each: {slot_id, label, bullets[], media{status,best_image_url,candidates[]}, done}
    if "agent_budget" not in st.session_state:
        st.session_state.agent_budget = GLOBAL_QUESTION_CAP
    if "finalized" not in st.session_state:
        st.session_state.finalized = False
    if "final_lines" not in st.session_state:
        st.session_state.final_lines = {}  # label -> line


def get_slot(slot_id: str) -> Optional[Dict[str, Any]]:
    for s in st.session_state.slots:
        if s.get("slot_id") == slot_id:
            return s
    return None


# =====================================
# Chat Loop Handling (Function Calling with correct ordering)
# =====================================

def append_assistant_with_tool_calls(message_obj) -> List[Dict[str, Any]]:
    """Append assistant message preserving tool_calls for API pairing. Returns normalized tool_calls list."""
    content = (message_obj.content or "").strip()
    tool_calls = getattr(message_obj, "tool_calls", None) or []

    assistant_msg: Dict[str, Any] = {
        "role": "assistant",
        "content": enforce_single_question(content) if content else ""
    }
    if tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}"
                }
            }
            for tc in tool_calls
        ]
    st.session_state.history.append(assistant_msg)

    if assistant_msg["content"].endswith("?"):
        st.session_state.agent_budget = max(0, st.session_state.agent_budget - 1)

    # Return simple list for iteration
    return [
        {
            "id": tc.id,
            "name": tc.function.name,
            "arguments": tc.function.arguments or "{}"
        }
        for tc in tool_calls
    ]


def handle_tool_call(tc_dict: Dict[str, Any]):
    name = tc_dict["name"]
    args = json.loads(tc_dict["arguments"]) if tc_dict.get("arguments") else {}

    if name == "MEDIA_REQUEST":
        slot_id = args.get("slot_id") or str(uuid.uuid4())[:8]
        intent = args.get("intent", "generic")
        query = args.get("query", "")
        constraints = args.get("constraints", {})

        slot = get_slot(slot_id)
        if slot is None:
            slot = {
                "slot_id": slot_id,
                "label": "",
                "bullets": [],
                "media": {"status": "pending", "best_image_url": "", "candidates": []},
                "done": False
            }
            st.session_state.slots.append(slot)

        media_payload = {
            "slot_id": slot_id,
            "intent": intent,
            "query": query,
            "constraints": constraints
        }
        media_result = media_agent_handle(media_payload)

        # Persist into slot
        slot["media"]["status"] = media_result.get("status", "none")
        slot["media"]["best_image_url"] = media_result.get("best_image_url", "")
        slot["media"]["candidates"] = media_result.get("candidates", [])

        # Post tool result message immediately after assistant(with tool_calls)
        st.session_state.history.append({
            "role": "tool",
            "tool_call_id": tc_dict["id"],
            "name": "MEDIA_REQUEST",
            "content": json.dumps(media_result)
        })

    elif name == "FINALIZE_CARD":
        notes = args.get("notes", [])
        # Map notes into slots (create if missing)
        for note in notes:
            label = (note.get("label") or "").strip()
            bullets = [b.strip() for b in note.get("bullets", []) if b and b.strip()]
            matched = None
            for s in st.session_state.slots:
                if not s.get("label"):
                    matched = s
                    break
            if matched is None:
                matched = {
                    "slot_id": str(uuid.uuid4())[:8],
                    "label": "",
                    "bullets": [],
                    "media": {"status": "pending", "best_image_url": "", "candidates": []},
                    "done": False
                }
                st.session_state.slots.append(matched)
            matched["label"] = label
            matched["bullets"] = bullets
            matched["done"] = True

        final_lines = finalize_card_from_notes(notes)
        st.session_state.final_lines = final_lines
        st.session_state.finalized = True

        st.session_state.history.append({
            "role": "tool",
            "tool_call_id": tc_dict["id"],
            "name": "FINALIZE_CARD",
            "content": json.dumps({"ok": True, "labels": list(final_lines.keys())})
        })


def run_chat_turn(user_text: Optional[str] = None):
    # Append user message if present
    if user_text:
        st.session_state.history.append({"role": "user", "content": user_text})

    # FIRST CALL
    res = chat_once(
        CHAT_MODEL,
        st.session_state.history,  # includes single system at index 0
        tools=CHAT_TOOLS,
        tool_choice="auto"
    )
    msg = res.choices[0].message
    pending = append_assistant_with_tool_calls(msg)  # append assistant (with tool_calls)

    # Loop through tool-calls until none remain
    # Each iteration: handle all tool calls → follow-up model call → append assistant w/ tool_calls
    loop_guard = 0
    while pending and loop_guard < 5:  # safety guard against runaway loops
        for tc in pending:
            handle_tool_call(tc)
        # Follow-up call so the agent can continue after tools
        res_next = chat_once(
            CHAT_MODEL,
            st.session_state.history,
            tools=CHAT_TOOLS,
            tool_choice="auto"
        )
        msg_next = res_next.choices[0].message
        pending = append_assistant_with_tool_calls(msg_next)
        loop_guard += 1

# =====================================
# UI RENDERING
# =====================================

def render_progress():
    filled = sum(1 for s in st.session_state.slots if s.get("label") and s.get("bullets"))
    st.progress(min(1.0, filled / MAX_SLOTS), text=f"Progress: {filled}/{MAX_SLOTS}")


def render_timeline():
    slots = st.session_state.slots
    if not slots:
        return
    cols = st.columns(min(len(slots), 4))
    for i, s in enumerate(slots[:4]):
        with cols[i]:
            label = s.get("label", "(pending)") or "(pending)"
            st.markdown(f"**{label}**")
            st.caption("status: " + (s.get("media", {}).get("status") or "pending"))
            best_url = (s.get("media", {}).get("best_image_url") or "").strip()
            if is_http_url(best_url):
                st.image(best_url, use_container_width=True)
            elif best_url:
                st.caption(f"Image note: {best_url}")


def render_chat():
    for m in st.session_state.history:
        if m["role"] == "system":
            continue
        if m["role"] == "tool":
            continue  # hide backend tool messages
        with st.chat_message(m["role"]):
            st.markdown(m["content"]) 


def render_final_card():
    if not st.session_state.finalized:
        return
    if not st.session_state.final_lines:
        return
    st.subheader("Your Expert Card")
    for s in st.session_state.slots[:MAX_SLOTS]:
        label = s.get("label", "").strip()
        if not label:
            continue
        line = st.session_state.final_lines.get(label, "").strip()
        if not line:
            continue
        best_url = (s.get("media", {}).get("best_image_url") or "").strip()
        cols = st.columns([1, 2])
        with cols[0]:
            if is_http_url(best_url):
                st.image(best_url, use_container_width=True)
            else:
                st.caption("(image pending)")
        with cols[1]:
            st.markdown(f"**{label}**")
            st.write(line)

# =====================================
# MAIN
# =====================================

st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
st.caption("Two-Agent setup: Chat Agent (interview/orchestration) + Media Agent (search/validation/generation)")

client = get_client()
init_state()

render_progress()
render_timeline()
render_chat()

# User input
user_text = st.chat_input("Type your answer…")
if user_text:
    run_chat_turn(user_text)
    st.rerun()

# If finalized, render card
render_final_card()

# Controls
c1, c2 = st.columns(2)
with c1:
    if st.button("🔄 Restart"):
        for k in ["history", "slots", "agent_budget", "finalized", "final_lines"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()
with c2:
    if st.button("🧹 Clear final"):
        st.session_state.finalized = False
        st.session_state.final_lines = {}
        st.rerun()

# Debug
if SHOW_DEBUG:
    with st.expander("Debug State"):
        st.json({
            "slots": st.session_state.slots,
            "finalized": st.session_state.finalized,
            "final_lines": st.session_state.final_lines
        })
