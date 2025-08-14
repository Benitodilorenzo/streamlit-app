# app.py — Expert Card Mini Agent (Streamlit + OpenAI Responses)
# - Orchestrator chat (no confirmations, at most 2 questions)
# - Autonomous decide: ask / start_research / continue
# - Background research pipeline while chat continues
# - Robust wrappers for Responses API changes (response_format/tools/models)

import os
import json
import io
import threading
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI
import svgwrite
from PIL import Image

# Optional high-quality SVG->PNG
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

# Core chat model (user requested GPT-5); add safe fallbacks
CHAT_MODEL_PRIMARY = os.getenv("DTBR_CHAT_MODEL", "gpt-5")
CHAT_MODEL_FALLBACKS = ["gpt-4o", "gpt-4o-mini"]

# Search/Tools model(s) for Responses + web_search
SEARCH_MODEL_PRIMARY = os.getenv("DTBR_SEARCH_MODEL", "gpt-5")
SEARCH_MODEL_FALLBACKS = ["gpt-4.1", "gpt-4o"]

# Limits / UX
MAX_QUESTIONS = 2  # at most two questions the bot should ask
SHOW_DEBUG = False  # set True to show control JSON

# =======================
# UTIL
# =======================
def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set (Streamlit → Settings → Secrets).")
    return OpenAI(api_key=OPENAI_API_KEY)

def ensure_session():
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_orchestrator_prompt()},
            {"role": "assistant",
             "content": ("Hi! I'll help you craft a tiny expert card.\n"
                         "First: which book has helped you professionally? "
                         "Please give the title (author optional).")}
        ]
    if "profile" not in st.session_state:
        st.session_state.profile = {"book": {"title": "", "why": "", "author_guess": ""}}
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "control" not in st.session_state:
        st.session_state.control = {"action": "ask", "assistant_visible": "", "ready": False}
    if "research" not in st.session_state:
        st.session_state.research = {"status": "idle", "error": "", "result": None}
    if "bg_thread" not in st.session_state:
        st.session_state.bg_thread = None

def svg_card(person_name: str, headline: str, verified: dict, summary: dict) -> str:
    dwg = svgwrite.Drawing(size=(SVG_W, SVG_H), profile="full")
    dwg.add(dwg.rect(insert=(0, 0), size=(SVG_W, SVG_H), fill="#ffffff"))
    # yellow panel
    dwg.add(dwg.rect((60, 120), (460, 400), rx=40, ry=40, fill=BRAND_YELLOW))
    # headline
    dwg.add(dwg.text(headline.upper(), insert=(90, 210), fill=TEXT_PRIMARY,
                     font_size="60px", font_weight="700", font_family="Montserrat, Arial"))
    # avatar placeholder circle
    dwg.add(dwg.circle(center=(680, 320), r=160, fill="#eeeeee"))
    # name
    dwg.add(dwg.text(person_name, insert=(60, 580), fill=TEXT_PRIMARY,
                     font_size="38px", font_weight="700", font_family="Montserrat, Arial"))
    # Book block
    dwg.add(dwg.text("Book", insert=(720, 120), fill=TEXT_PRIMARY,
                     font_size="30px", font_weight="700", font_family="Montserrat, Arial"))
    title = (verified.get("title") or "").strip()
    author = (verified.get("author") or "").strip()
    if title:
        dwg.add(dwg.text(title, insert=(720, 160), fill=TEXT_PRIMARY, font_size="24px"))
    if author:
        dwg.add(dwg.text(author, insert=(720, 190), fill=TEXT_PRIMARY, font_size="22px"))
    # summary (naive wrap)
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
    img = Image.new("RGB", (SVG_W, SVG_H), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# =======================
# AGENT PROMPTS
# =======================
def system_orchestrator_prompt() -> str:
    """
    The single system prompt guiding the conversation.
    - Default English; mirror user's language if obviously not English.
    - Collect exactly two required fields: book.title and book.why (1–2 sentences).
    - Ask at most TWO concise questions, one at a time.
    - Do NOT ask for confirmations the user already implied.
    - Do NOT wait for confirmation to start research; the app/agent may start it in the background.
    - Keep replies short and friendly.
    """
    return (
        "You are an interviewing assistant for a tiny 'Expert Card'.\n"
        "LANGUAGE:\n"
        "- Default to English; mirror user's language if clearly not English.\n"
        "OBJECTIVE (TWO FIELDS):\n"
        "- Collect exactly two fields: book.title, book.why (1–2 sentences).\n"
        "INTERACTION RULES:\n"
        "- Ask at most TWO concise questions total. One at a time.\n"
        "- If the user already supplied both fields, do NOT ask for confirmation.\n"
        "- Avoid repeating information already given. Be brief, friendly, precise.\n"
        "- Do not mention background research explicitly.\n"
    )

def control_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "assistant_visible": {"type": "string", "description": "Next assistant message to show the user (short)."},
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
            "action": {"type": "string", "enum": ["ask", "start_research", "continue"]},
            "ready_to_research": {"type": "boolean"}
        },
        "required": ["assistant_visible", "profile_partial", "action", "ready_to_research"]
    }

def controller_payload(history: List[Dict[str, str]], question_count: int) -> Dict[str, Any]:
    """
    A single JSON-schema call that:
    - writes the next assistant message (assistant_visible),
    - merges extracted profile fields (profile_partial),
    - decides action: ask | start_research | continue,
    - sets ready_to_research.
    """
    schema = control_schema()
    return {
        "model": CHAT_MODEL_PRIMARY,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "orchestrator_control", "schema": schema, "strict": True}
        },
        "input": [
            {"role": "system", "content":
                "From the conversation, produce the next short assistant message and control flags. "
                "Collect only two fields: book.title and book.why. "
                f"The bot has asked {question_count} question(s) so far; do not exceed two total. "
                "If both fields are present with reasonable detail, set action='start_research' and ready_to_research=true. "
                "Do NOT ask for confirmations. If something is missing, set action='ask', ready_to_research=false, and ask exactly one concise question. "
                "If everything is okay but no more questions are needed, set action='continue' and produce a brief follow-up message. "
                "Return ONLY a single JSON object."
            },
            *history[-12:]
        ]
    }

def research_payload(model: str, title: str, author_guess: str) -> Dict[str, Any]:
    return {
        "model": model,
        "tools": [{"type": "web_search"}],
        "response_format": {"type": "json_object"},
        "input": [
            {"role": "system", "content":
                "Use the web_search tool to find book data. "
                "Return JSON with 'candidates' (up to 5), each: "
                "title (str), authors (array), cover_url (str), info_url (str), source (str). "
                "Prefer publisher sites, Google Books, Open Library. Respond with ONLY JSON."},
            {"role": "user", "content": json.dumps({"book_title": title, "author_guess": author_guess}, separators=(",", ":"))}
        ]
    }

def verifier_payload(user_claim: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        "model": CHAT_MODEL_PRIMARY,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "book_verification", "schema": schema, "strict": True}},
        "input": [
            {"role": "system",
             "content": "Pick the best candidate matching the user's claim. If unsure, set status=not_found. Output STRICT JSON only."},
            {"role": "user", "content": json.dumps({"user_claim": user_claim, "candidates": candidates}, separators=(",", ":"))}
        ]
    }

def cover_audit_payload(verified: Dict[str, Any]) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "cover_is_plausible": {"type": "boolean"},
            "reason": {"type": "string"}
        },
        "required": ["cover_is_plausible", "reason"]
    }
    return {
        "model": CHAT_MODEL_PRIMARY,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "cover_audit", "schema": schema, "strict": True}},
        "input": [
            {"role": "system", "content":
                "Audit whether the chosen cover_url plausibly matches the book title/author. "
                "Consider typical covers, editions, and common mismatches. Output STRICT JSON."},
            {"role": "user", "content": json.dumps(verified, separators=(",", ":"))}
        ]
    }

def summary_payload(why: str, title: str, author: str) -> Dict[str, Any]:
    schema = {"type": "object",
              "properties": {"book_100w": {"type": "string"}, "one_liner": {"type": "string"}},
              "required": ["book_100w", "one_liner"]}
    return {
        "model": CHAT_MODEL_PRIMARY,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "summaries", "schema": schema, "strict": True}},
        "input": [
            {"role": "system", "content": "Summarize the user's reason in ~100 words and a one-liner (<=18 words). No inventions. STRICT JSON."},
            {"role": "user", "content": json.dumps({"user_book_why": why, "book_title": title, "book_author": author}, separators=(",", ":"))}
        ]
    }

# =======================
# OPENAI WRAPPERS (robust)
# =======================
def responses_create(client: OpenAI, **kwargs):
    """
    Robust wrapper: try as-is; if SDK complains about unknown kwargs (e.g. response_format),
    drop them and retry. Also allow model fallbacks by raising to caller to handle.
    """
    try:
        return client.responses.create(**kwargs)
    except TypeError:
        cleaned = dict(kwargs)
        cleaned.pop("response_format", None)
        cleaned.pop("tools", None)  # in case older SDK
        return client.responses.create(**cleaned)

def responses_stream(client: OpenAI, **kwargs):
    """
    Robust stream wrapper with same dropping behavior.
    """
    try:
        return client.responses.stream(**kwargs)
    except TypeError:
        cleaned = dict(kwargs)
        cleaned.pop("response_format", None)
        cleaned.pop("tools", None)
        return client.responses.stream(**cleaned)

# =======================
# LLM CALLS
# =======================
def controller_decide(client: OpenAI, history: List[Dict[str, str]], question_count: int,
                      model_list: List[str]) -> Dict[str, Any]:
    """
    Try primary + fallbacks for the controller call.
    Returns dict with keys: assistant_visible, action, ready_to_research, profile_partial
    """
    last_err = None
    for m in model_list:
        payload = controller_payload(history, question_count)
        payload["model"] = m
        try:
            r = responses_create(client, **payload)
            text = getattr(r, "output_text", "")
            if not text and getattr(r, "output", None):
                blk = r.output[0]
                if blk and blk.get("content") and blk["content"][0].get("text"):
                    text = blk["content"][0]["text"]
            data = json.loads(text)
            return data
        except Exception as e:
            last_err = e
            continue
    # fallback minimal
    if SHOW_DEBUG:
        st.warning(f"Controller fallback due to error: {last_err}")
    return {"assistant_visible": "Thanks! Please tell me briefly why this book helped you.",
            "profile_partial": {}, "action": "ask", "ready_to_research": False}

def run_research_pipeline(client: OpenAI, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background pipeline:
      search (with tool) → verify → cover audit → summary
    Uses model fallbacks if the first fails (e.g., tool unsupported).
    """
    title = (profile["book"].get("title") or "").strip()
    author_guess = (profile["book"].get("author_guess") or "").strip()

    # 1) SEARCH with tool model fallbacks
    last_err = None
    r1 = None
    for m in [SEARCH_MODEL_PRIMARY, *SEARCH_MODEL_FALLBACKS]:
        try:
            r1 = responses_create(client, **research_payload(m, title, author_guess))
            break
        except Exception as e:
            last_err = e
            continue
    if r1 is None:
        raise last_err or RuntimeError("Search failed: no suitable model for web_search tool.")

    ctext = getattr(r1, "output_text", "") or (r1.output[0]["content"][0]["text"] if getattr(r1, "output", None) else "")
    try:
        candidates = json.loads(ctext).get("candidates", [])
    except Exception:
        candidates = []

    # 2) VERIFY (structured)
    r2 = responses_create(client, **verifier_payload(profile["book"], candidates))
    vtext = getattr(r2, "output_text", "") or (r2.output[0]["content"][0]["text"] if getattr(r2, "output", None) else "")
    try:
        verified = json.loads(vtext)
    except Exception:
        verified = {
            "status": "not_found", "title": title, "author": author_guess,
            "cover_url": "", "info_url": "", "citations": [], "verification": "Parse failed"
        }

    # 3) COVER AUDIT (structured)
    r3 = responses_create(client, **cover_audit_payload(verified))
    atext = getattr(r3, "output_text", "") or (r3.output[0]["content"][0]["text"] if getattr(r3, "output", None) else "")
    try:
        audit = json.loads(atext)
    except Exception:
        audit = {"cover_is_plausible": True, "reason": "fallback"}

    # 4) SUMMARY (structured)
    r4 = responses_create(client, **summary_payload(profile["book"].get("why", ""), verified.get("title", ""), verified.get("author", "")))
    stext = getattr(r4, "output_text", "") or (r4.output[0]["content"][0]["text"] if getattr(r4, "output", None) else "")
    try:
        summary = json.loads(stext)
    except Exception:
        summary = {"book_100w": profile["book"].get("why", ""), "one_liner": ""}

    return {"verified": verified, "summary": summary, "audit": audit, "candidates": candidates}

# =======================
# BACKGROUND EXECUTION
# =======================
def start_background_research(client: OpenAI, profile_snapshot: Dict[str, Any]):
    if st.session_state.research["status"] in ("running", "done"):
        return  # already running/done
    st.session_state.research = {"status": "running", "error": "", "result": None}

    def _worker():
        try:
            result = run_research_pipeline(client, profile_snapshot)
            st.session_state.research = {"status": "done", "error": "", "result": result}
        except Exception as e:
            st.session_state.research = {"status": "error", "error": str(e), "result": None}

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    st.session_state.bg_thread = t

# =======================
# STREAMLIT APP
# =======================
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="centered")
st.title("Dein Kurz-Steckbrief")
st.caption("Mini agent: interview → autonomous decide → background research → SVG/PNG")

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

# --- Chat input & control ---
user_text = st.chat_input("Type your answer…")
if user_text:
    # append user message
    st.session_state.messages.append({"role": "user", "content": user_text})

    if agent_ready:
        # CONTROLLER decides next visible text + action; also extracts fields
        ctrl = controller_decide(client, st.session_state.messages, st.session_state.question_count,
                                 [CHAT_MODEL_PRIMARY, *CHAT_MODEL_FALLBACKS])

        # merge profile partial
        partial = ctrl.get("profile_partial") or {}
        bookp = partial.get("book") or {}
        for k, v in bookp.items():
            if isinstance(v, str) and not v.strip():
                continue
            st.session_state.profile["book"][k] = v

        # visible assistant message
        visible = ctrl.get("assistant_visible") or ""
        if visible.strip():
            st.session_state.messages.append({"role": "assistant", "content": visible})
            # count only if it *is* a question (very rough heuristic: ends with '?')
            if visible.strip().endswith("?"):
                st.session_state.question_count += 1

        # action handling
        action = ctrl.get("action") or "ask"
        ready = bool(ctrl.get("ready_to_research", False))
        st.session_state.control = {"action": action, "assistant_visible": visible, "ready": ready}

        # If agent says to start research, kick off background worker (no confirmation)
        if ready and action == "start_research":
            start_background_research(client, json.loads(json.dumps(st.session_state.profile)))  # snapshot

        if SHOW_DEBUG:
            with st.expander("DEBUG control JSON"):
                st.code(json.dumps(ctrl, indent=2))
    else:
        st.session_state.messages.append({"role": "assistant",
                                          "content": "(Demo mode — set OPENAI_API_KEY in Streamlit Secrets.)"})
    st.rerun()

# --- Background research status & result ---
rs = st.session_state.research
if rs["status"] == "running":
    st.info("🔎 Research is running in the background… you can continue chatting.")
elif rs["status"] == "error":
    st.error(f"Research failed: {rs['error']}")
elif rs["status"] == "done" and rs["result"]:
    bundle = rs["result"]
    verified, summary = bundle["verified"], bundle["summary"]
    svg = svg_card("Member", "EXPERT PICKS", verified, summary)
    st.session_state["last_svg"] = svg
    st.success("✅ Expert card is ready.")
    st.download_button("Download SVG", data=svg.encode("utf-8"),
                       file_name="expert_card.svg", mime="image/svg+xml")
    png = png_from_svg(svg)
    st.download_button("Download PNG", data=png,
                       file_name="expert_card.png", mime="image/png")

# --- Controls ---
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🔄 Restart interview"):
        for k in ["messages", "profile", "question_count", "control", "research", "bg_thread", "last_svg"]:
            if k in st.session_state:
                del st.session_state[k]
        ensure_session()
        st.rerun()
with c2:
    if st.session_state.get("last_svg"):
        st.download_button("⬇️ Download last SVG again",
                           data=st.session_state["last_svg"].encode("utf-8"),
                           file_name="expert_card.svg", mime="image/svg+xml")
with c3:
    if SHOW_DEBUG and st.button("Show profile JSON"):
        st.code(json.dumps(st.session_state.profile, indent=2))
