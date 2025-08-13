import os
import json
import time
import io
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI
import svgwrite
from PIL import Image

# Optional high-quality SVG->PNG (recommended; add to requirements.txt)
try:
    import cairosvg  # type: ignore
    HAS_CAIROSVG = True
except Exception:
    HAS_CAIROSVG = False

# =======================
# CONFIG
# =======================
APP_TITLE = "Expert Card – Mini Agent"
SVG_W, SVG_H = 1200, 1200
BRAND_YELLOW = "#fcc814"
TEXT_PRIMARY = "#111111"
TEXT_SECONDARY = "#333333"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("DTBR_CHAT_MODEL", "gpt-4o-mini")
SEARCH_MODEL = os.getenv("DTBR_SEARCH_MODEL", "gpt-4o-mini-search-preview")  # supports web_search tool

# =======================
# UTIL
# =======================
def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set (Streamlit → Settings → Secrets).")
    return OpenAI(api_key=OPENAI_API_KEY)

def ensure_session():
    if "messages" not in st.session_state:
        # conversation history including a system message
        st.session_state.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_orchestrator_prompt()},
            {"role": "assistant",
             "content": "Hi! I'll help you craft a tiny expert card. "
                        "First: which book has helped you professionally? "
                        "Please give the title (author optional)."},
        ]
    if "profile" not in st.session_state:
        st.session_state.profile = {
            "book": {"title": "", "why": "", "author_guess": ""}
        }
    if "controller" not in st.session_state:
        st.session_state.controller = {"ready_to_research": False, "next_question": ""}

def svg_card(person_name: str, headline: str, verified: dict, summary: dict) -> str:
    dwg = svgwrite.Drawing(size=(SVG_W, SVG_H), profile="full")
    dwg.add(dwg.rect(insert=(0, 0), size=(SVG_W, SVG_H), fill="#ffffff"))
    # yellow panel
    dwg.add(dwg.rect((60, 120), (460, 400), rx=40, ry=40, fill=BRAND_YELLOW))
    # headline
    dwg.add(dwg.text(headline.upper(), insert=(90, 210), fill=TEXT_PRIMARY,
                     font_size="60px", font_weight="700", font_family="Montserrat, Arial"))
    # avatar placeholder circle (we keep it simple for now)
    dwg.add(dwg.circle(center=(680, 320), r=160, fill="#eeeeee"))
    # name
    dwg.add(dwg.text(person_name, insert=(60, 580), fill=TEXT_PRIMARY,
                     font_size="38px", font_weight="700", font_family="Montserrat, Arial"))
    # Book block
    dwg.add(dwg.text("Book", insert=(720, 120), fill=TEXT_PRIMARY,
                     font_size="30px", font_weight="700", font_family="Montserrat, Arial"))
    title = verified.get("title", "") or ""
    author = verified.get("author", "") or ""
    dwg.add(dwg.text(title, insert=(720, 160), fill=TEXT_PRIMARY, font_size="24px"))
    if author:
        dwg.add(dwg.text(author, insert=(720, 190), fill=TEXT_PRIMARY, font_size="22px"))
    # summary text (wrap naive)
    text_y = 230
    para = (summary.get("book_100w", "") or "").strip()
    line, lines = "", []
    for w in para.split():
        test = (w if not line else f"{line} {w}")
        if len(test) > 60:
            lines.append(line); line = w
        else:
            line = test
    if line: lines.append(line)
    for ln in lines[:8]:
        dwg.add(dwg.text(ln, insert=(720, text_y), fill=TEXT_SECONDARY, font_size="20px"))
        text_y += 28
    # dots
    for i in range(12):
        x = 560 + (i * 45)
        y = 520 + (30 if (i % 2) else -10)
        dwg.add(dwg.circle(center=(x, y), r=6, fill="#8ad0a6"))
    return dwg.tostring()

def png_from_svg(svg_text: str) -> bytes:
    if HAS_CAIROSVG:
        return cairosvg.svg2png(bytestring=svg_text.encode("utf-8"),
                                output_width=SVG_W, output_height=SVG_H)
    # minimal placeholder if cairosvg missing
    img = Image.new("RGB", (SVG_W, SVG_H), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# =======================
# AGENT PROMPTS (ROLES)
# =======================
def system_orchestrator_prompt() -> str:
    """
    This is the single system prompt for the conversation. It sets policy & tone.
    """
    return (
        "You are an interviewing assistant for a tiny 'Expert Card'.\n"
        "GENERAL:\n"
        "- Default language is English. If the user clearly writes in another language, reply in that language.\n"
        "- Ask at most TWO concise questions, one at a time, to collect:\n"
        "  1) book.title (required)\n"
        "  2) book.why   (required, 1–2 sentences)\n"
        "- If the user voluntarily provides both pieces, do not ask more—confirm and proceed.\n"
        "- Be brief, friendly, and precise. No long paragraphs.\n"
        "CONTROL:\n"
        "- After each user answer, the app will call a separate controller to extract fields and decide readiness.\n"
        "- You only focus on natural conversation. Keep messages short and clear.\n"
    )

def controller_prompt(history: List[Dict[str, str]], current_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the controller input for the JSON-schema extractor/decider.
    """
    return {
        "model": CHAT_MODEL,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "interview_control",
                "schema": {
                    "type": "object",
                    "properties": {
                        "profile_partial": {
                            "type": "object",
                            "properties": {
                                "book": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "why": {"type": "string"},
                                        "author_guess": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "ready_to_research": {"type": "boolean"},
                        "next_question": {"type": "string", "description": "Short, natural next question in the user's language; empty if none."}
                    },
                    "required": ["profile_partial", "ready_to_research", "next_question"]
                },
                "strict": True
            }
        },
        "input": [
            {
                "role": "system",
                "content": (
                    "From the conversation so far, extract the fields and decide if we have enough to start research.\n"
                    "We need exactly two required fields: book.title and book.why (1–2 sentences). "
                    "If both are reasonably present, set ready_to_research=true. "
                    "If not ready, propose next_question (very short). "
                    "Default language English; mirror user's language if obvious."
                )
            },
            *history[-12:]
        ]
    }

def research_prompt(title: str, author_guess: str) -> Dict[str, Any]:
    """
    Responses call that uses web_search tool to fetch candidates for the book.
    """
    return {
        "model": SEARCH_MODEL,
        "tools": [{"type": "web_search"}],
        "response_format": {"type": "json_object"},
        "input": [
            {"role": "system", "content":
                "Use the web_search tool to find book data. "
                "Return JSON with 'candidates' (up to 5), each having: "
                "title (str), authors (array), cover_url (str), info_url (str), source (str). "
                "Prefer publisher sites, Google Books, Open Library. If uncertain, include your best plausible options."},
            {"role": "user", "content": json.dumps({"book_title": title, "author_guess": author_guess})}
        ]
    }

def verifier_prompt(user_claim: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "not_found"]},
            "title": {"type": "string"},
            "author": {"type": "string"},
            "cover_url": {"type": "string"},
            "info_url": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
            "verification": {"type": "string"}
        },
        "required": ["status", "title", "author", "cover_url", "info_url", "citations", "verification"]
    }
    return {
        "model": CHAT_MODEL,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "book_verification", "schema": schema, "strict": True}},
        "input": [
            {"role": "system",
             "content": "Pick the best candidate matching the user's claim. If unsure, set status=not_found. Output strict JSON only."},
            {"role": "user", "content": json.dumps({"user_claim": user_claim, "candidates": candidates}, separators=(",", ":"))}
        ]
    }

def summary_prompt(why: str, title: str, author: str) -> Dict[str, Any]:
    schema = {"type": "object",
              "properties": {"book_100w": {"type": "string"}, "one_liner": {"type": "string"}},
              "required": ["book_100w", "one_liner"]}
    return {
        "model": CHAT_MODEL,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "summaries", "schema": schema, "strict": True}},
        "input": [
            {"role": "system", "content": "Summarize the user's reason in ~100 words and a one-liner (<=18 words). No inventions."},
            {"role": "user", "content": json.dumps({"user_book_why": why, "book_title": title, "book_author": author}, separators=(",", ":"))}
        ]
    }

# =======================
# LLM CALLS
# =======================
def stream_assistant_reply(client: OpenAI, history: list[dict[str, str]]) -> str:
    """
    Streamt die Assistenten-Antwort korrekt:
    - Context-Manager benutzen
    - über .events() iterieren
    """
    chunks: list[str] = []
    with st.chat_message("assistant"):
        ph = st.empty()
        # WICHTIG: als Context-Manager + .events()
        with client.responses.stream(model=CHAT_MODEL, input=history) as stream:
            for event in stream.events():
                if event.type == "response.output_text.delta":
                    chunks.append(event.delta)
                    ph.markdown("".join(chunks))
            # optional: Finale Response abrufen (falls du sie brauchst)
            _final = stream.get_final_response()
    return "".join(chunks).strip()


def controller_decide(client: OpenAI, history: List[Dict[str, str]], current_profile: Dict[str, Any]) -> Dict[str, Any]:
    payload = controller_prompt(history, current_profile)
    r = client.responses.create(**payload)
    text = getattr(r, "output_text", "")
    if not text and getattr(r, "output", None):
        blk = r.output[0]
        if blk and blk.get("content") and blk["content"][0].get("text"):
            text = blk["content"][0]["text"]
    try:
        data = json.loads(text)
    except Exception:
        data = {"profile_partial": {}, "ready_to_research": False, "next_question": ""}
    # merge partial into profile
    prof = current_profile.copy()
    partial = data.get("profile_partial") or {}
    for k, v in partial.get("book", {}).items():
        if isinstance(v, str) and not v.strip():
            continue
        prof["book"][k] = v
    return {"profile": prof,
            "ready_to_research": bool(data.get("ready_to_research", False)),
            "next_question": data.get("next_question", "")[:240]}

def run_research_pipeline(client: OpenAI, profile: Dict[str, Any]) -> Dict[str, Any]:
    title = profile["book"].get("title", "")
    author_guess = profile["book"].get("author_guess", "")
    # 1) search
    r1 = client.responses.create(**research_prompt(title, author_guess))
    content = getattr(r1, "output_text", "") or (r1.output[0]["content"][0]["text"] if getattr(r1, "output", None) else "")
    try:
        candidates = json.loads(content).get("candidates", [])
    except Exception:
        candidates = []
    # 2) verify
    r2 = client.responses.create(**verifier_prompt(profile["book"], candidates))
    v_text = getattr(r2, "output_text", "") or (r2.output[0]["content"][0]["text"] if getattr(r2, "output", None) else "")
    try:
        verified = json.loads(v_text)
    except Exception:
        verified = {"status": "not_found", "title": title, "author": author_guess, "cover_url": "", "info_url": "", "citations": [], "verification": "Parse failed"}
    # 3) summary
    r3 = client.responses.create(**summary_prompt(profile["book"].get("why", ""), verified.get("title", ""), verified.get("author", "")))
    s_text = getattr(r3, "output_text", "") or (r3.output[0]["content"][0]["text"] if getattr(r3, "output", None) else "")
    try:
        summary = json.loads(s_text)
    except Exception:
        summary = {"book_100w": profile["book"].get("why", ""), "one_liner": ""}
    return {"verified": verified, "summary": summary}

# =======================
# STREAMLIT APP
# =======================
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="centered")
st.title("Dein Kurz-Steckbrief")
st.caption("Mini agent: interview → decide → research/verify → SVG/PNG")

ensure_session()

# Render history (skip system)
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Client ready?
client = None
agent_ready = True
try:
    client = get_client()
except Exception as e:
    agent_ready = False
    st.warning(f"OpenAI key missing: {e}")

# Chat input
user_text = st.chat_input("Type your answer…")
if user_text:
    # append user msg
    st.session_state.messages.append({"role": "user", "content": user_text})
    # ask LLM to reply
    if agent_ready:
        reply = stream_assistant_reply(client, st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        # controller: extract/decide/next question
        ctrl = controller_decide(client, st.session_state.messages, st.session_state.profile)
        st.session_state.profile = ctrl["profile"]
        st.session_state.controller = {"ready_to_research": ctrl["ready_to_research"], "next_question": ctrl["next_question"]}
        # if not ready and we have a concrete next question, show it immediately
        if not ctrl["ready_to_research"] and ctrl["next_question"]:
            st.session_state.messages.append({"role": "assistant", "content": ctrl["next_question"]})
    else:
        st.session_state.messages.append({"role": "assistant", "content": "(Demo mode — set OPENAI_API_KEY in Streamlit Secrets.)"})
    st.rerun()

# If controller says we're ready, auto-signal next step
if st.session_state.controller.get("ready_to_research"):
    st.info("✅ I have enough info. Starting research & validation…")
    if agent_ready:
        with st.status("Searching & validating…", expanded=True) as status:
            bundle = run_research_pipeline(client, st.session_state.profile)
            verified, summary = bundle["verified"], bundle["summary"]
            # render SVG
            person_name = "Member"
            svg = svg_card(person_name, "EXPERT PICKS", verified, summary)
            st.session_state["last_svg"] = svg

            st.success("Expert card ready.")
            st.download_button("Download SVG", data=svg.encode("utf-8"),
                               file_name="expert_card.svg", mime="image/svg+xml")
            png = png_from_svg(svg)
            st.download_button("Download PNG", data=png,
                               file_name="expert_card.png", mime="image/png")
            status.update(label="Done.", state="complete", expanded=False)

# Controls
c1, c2 = st.columns(2)
with c1:
    if st.button("🔄 Restart interview"):
        for k in ["messages", "profile", "controller", "last_svg"]:
            if k in st.session_state:
                del st.session_state[k]
        ensure_session()
        st.rerun()
with c2:
    if st.session_state.get("last_svg"):
        st.download_button("⬇️ Download last SVG again",
                           data=st.session_state["last_svg"].encode("utf-8"),
                           file_name="expert_card.svg", mime="image/svg+xml")
