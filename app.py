import os
import time
import hmac
import hashlib
import base64
import json
import io
from datetime import datetime, timezone

import streamlit as st
from openai import OpenAI
import svgwrite
from PIL import Image

# ========= CONFIG =========
APP_TITLE = "Datentreiber Steckbrief (Streamlit)"
SVG_WIDTH = 1200
SVG_HEIGHT = 1200
BRAND_YELLOW = "#fcc814"
PRIMARY_TEXT = "#111111"
SECONDARY_TEXT = "#333333"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("DTBR_CHAT_MODEL", "gpt-4o-mini")
SEARCH_MODEL = os.getenv("DTBR_SEARCH_MODEL", "gpt-4o-mini-search-preview")  # web_search tool
# Optional: gemeinsames Geheimnis mit WP für Token-Verifizierung
EMBED_SHARED_SECRET = os.getenv("DTBR_EMBED_SHARED_SECRET", "")  # z.B. aus WP SALT abgeleitet

# ========= HELPERS =========
def verify_wp_token(token: str) -> dict:
    """
    Erwartet ein base64-JSON mit fields: {user_id, display_name, email, ts, hmac}
    hmac = HMAC_SHA256(secret, base64url(headerless_payload))
    """
    if not token or not EMBED_SHARED_SECRET:
        return {}
    try:
        raw = base64.urlsafe_b64decode(token + "==").decode("utf-8")
        data = json.loads(raw)
        sig = data.pop("hmac", "")
        msg = base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode("utf-8"))
        expect = hmac.new(EMBED_SHARED_SECRET.encode("utf-8"), msg, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expect, sig):
            return {}
        # 10-Minuten-Window
        if abs(time.time() - int(data.get("ts", 0))) > 600:
            return {}
        return data
    except Exception:
        return {}

def build_agent_prompt():
    system = (
        "You are a helpful research & validation agent. "
        "Interview the user to collect a short expert profile: a book recommendation with a concise reason, "
        "optionally a favorite tool and a role model. "
        "Use the web_search tool to find and validate the book (title, author, credible source, cover image). "
        "Always return concise, structured content; ask one question at a time. "
        "When the user asks to 'finish' or clicks the finish button (simulated), summarize cleanly. "
        "Avoid hallucinations; if unsure about data, explicitly say so and propose alternatives."
    )
    return system

def run_web_search(client: OpenAI, title: str, author_guess: str):
    """
    Uses Responses API with web_search tool to get a few candidates.
    Returns list of dicts with title, authors, cover_url, info_url, source.
    """
    payload = {
        "model": SEARCH_MODEL,
        "tools": [{"type": "web_search"}],
        "response_format": {"type": "json_object"},
        "input": [
            {"role": "system", "content":
                "You use the web_search tool to find book data. "
                "Return up to 5 candidates as JSON array under 'candidates' with fields: "
                "title (str), authors (array), cover_url (str), info_url (str), source (str). "
                "Prefer publisher pages, Google Books, Open Library, Wikipedia (as fallback)."
            },
            {"role": "user", "content": json.dumps({
                "book_title": title,
                "author_guess": author_guess
            })}
        ]
    }
    r = client.responses.create(**payload)
    # Try to parse JSON content
    content = ""
    if hasattr(r, "output_text"):
        content = r.output_text
    elif getattr(r, "output", None):
        blk = r.output[0]
        if blk and blk.get("content") and blk["content"][0].get("text"):
            content = blk["content"][0]["text"]

    try:
        obj = json.loads(content) if content else {}
        return obj.get("candidates", [])
    except Exception:
        return []

def pick_best_book(client: OpenAI, user_claim: dict, candidates: list) -> dict:
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
    payload = {
        "model": CHAT_MODEL,
        "response_format": {"type": "json_schema", "json_schema": {"name": "book_verification", "schema": schema, "strict": True}},
        "input": [
            {"role": "system", "content": "From the given candidates, pick the best match to the user's claim. If uncertain, set status=not_found. Output strict JSON only."},
            {"role": "user", "content": json.dumps({"user_claim": user_claim, "candidates": candidates}, separators=(",", ":"))}
        ]
    }
    r = client.responses.create(**payload)
    content = ""
    if hasattr(r, "output_text"):
        content = r.output_text
    elif getattr(r, "output", None):
        blk = r.output[0]
        if blk and blk.get("content") and blk["content"][0].get("text"):
            content = blk["content"][0]["text"]
    try:
        return json.loads(content)
    except Exception:
        return {"status": "not_found", "title": user_claim.get("title", "—"), "author": user_claim.get("author_guess", ""), "cover_url": "", "info_url": "", "citations": [], "verification": "Parsing failed."}

def summarize_reason(client: OpenAI, book_why: str, title: str, author: str) -> dict:
    schema = {"type":"object","properties":{"book_100w":{"type":"string"},"one_liner":{"type":"string"}},"required":["book_100w","one_liner"]}
    payload = {
        "model": CHAT_MODEL,
        "response_format": {"type":"json_schema","json_schema":{"name":"summaries","schema":schema,"strict":True}},
        "input": [
            {"role":"system","content":"Summarize the user's reason into ~100 words and a one-liner (<=18 words). No inventions."},
            {"role":"user","content": json.dumps({"user_book_why": book_why, "book_title": title, "book_author": author}, separators=(",", ":"))}
        ]
    }
    r = client.responses.create(**payload)
    content = ""
    if hasattr(r, "output_text"):
        content = r.output_text
    elif getattr(r, "output", None):
        blk = r.output[0]
        if blk and blk.get("content") and blk["content"][0].get("text"):
            content = blk["content"][0]["text"]
    try:
        return json.loads(content)
    except Exception:
        return {"book_100w": book_why, "one_liner": ""}

def svg_card(person_name: str, tagline: str, headline: str, verified: dict, summary: dict, avatar_url: str = "") -> str:
    dwg = svgwrite.Drawing(size=(SVG_WIDTH, SVG_HEIGHT), profile="full")
    # bg
    dwg.add(dwg.rect(insert=(0,0), size=(SVG_WIDTH, SVG_HEIGHT), fill="#ffffff"))
    # yellow panel
    dwg.add(dwg.rect(insert=(60,120), size=(460,400), rx=40, ry=40, fill=BRAND_YELLOW))
    # headline
    dwg.add(dwg.text(headline.upper(), insert=(90,210), fill=PRIMARY_TEXT, font_size="60px", font_weight="700", font_family="Montserrat, Arial"))
    # (avatar as circle outline + image mask is complex in raw SVG; we keep a placeholder circle + optional raster overlay later)
    dwg.add(dwg.circle(center=(680,320), r=160, fill="#eee"))
    # name + tagline
    dwg.add(dwg.text(person_name, insert=(60,580), fill=PRIMARY_TEXT, font_size="38px", font_weight="700", font_family="Montserrat, Arial"))
    if tagline:
        dwg.add(dwg.text(tagline, insert=(60,620), fill=SECONDARY_TEXT, font_size="26px", font_family="Montserrat, Arial"))

    # Book block
    dwg.add(dwg.text("Book", insert=(720,120), fill=PRIMARY_TEXT, font_size="30px", font_weight="700", font_family="Montserrat, Arial"))
    title = verified.get("title","")
    author = verified.get("author","")
    dwg.add(dwg.text(title, insert=(720,160), fill=PRIMARY_TEXT, font_size="24px", font_family="Montserrat, Arial"))
    if author:
        dwg.add(dwg.text(author, insert=(720,190), fill=PRIMARY_TEXT, font_size="22px", font_family="Montserrat, Arial"))

    # summary (wrap naive)
    text_y = 230
    para = (summary.get("book_100w","") or "").strip()
    wrap = []
    cur = ""
    for w in para.split():
        if len(cur) + 1 + len(w) > 60:
            wrap.append(cur)
            cur = w
        else:
            cur = (w if not cur else cur + " " + w)
    if cur:
        wrap.append(cur)
    for line in wrap[:8]:
        dwg.add(dwg.text(line, insert=(720, text_y), fill=SECONDARY_TEXT, font_size="20px", font_family="Arial"))
        text_y += 28

    # dots deco
    for i in range(12):
        x = 560 + (i * 45)
        y = 520 + (30 if (i % 2) else -10)
        dwg.add(dwg.circle(center=(x,y), r=6, fill="#8ad0a6"))

    return dwg.tostring()

def rasterize_svg(svg_text: str) -> bytes:
    """Convert SVG -> PNG (very lightweight via Pillow by rendering blank and overlay? For real quality use cairosvg.)"""
    # Minimal fallback – empfehle in Produktion 'cairosvg' zu nutzen. Hier halten wir Abhängigkeiten schlank.
    try:
        import cairosvg  # optional if installed
        return cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), output_width=SVG_WIDTH, output_height=SVG_HEIGHT)
    except Exception:
        # Fallback: generate blank placeholder if cairosvg not available
        img = Image.new("RGB", (SVG_WIDTH, SVG_HEIGHT), (255,255,255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "profile" not in st.session_state:
        st.session_state.profile = {
            "headline": "EXPERT PICKS",
            "book": {"title": "", "why": "", "author_guess": ""},
            "tool": {"name": "", "why": ""},
            "role_model": {"name": "", "why": ""}
        }
    if "verified" not in st.session_state:
        st.session_state.verified = {}
    if "summary" not in st.session_state:
        st.session_state.summary = {}
    if "agent_ready" not in st.session_state:
        st.session_state.agent_ready = False

# ========= UI =========
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="centered")
init_session()

# Optional: eingehendes Token prüfen (für WP-Embed SSO)
params = st.query_params
wp_claim = {}
if "token" in params:
    wp_claim = verify_wp_token(params.get("token"))
user_display = wp_claim.get("display_name", "") if wp_claim else ""

st.title("Dein Kurz-Steckbrief")
st.caption("Streamlit-App (Chat • Recherche • Validierung • SVG/PNG-Export)")

# Chat display
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Agent Setup (einmalig)
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
    st.session_state.agent_ready = True
else:
    st.warning("OPENAI_API_KEY nicht gesetzt – bitte als Umgebungsvariable hinterlegen.")

def ask_agent(question: str):
    """Minimalistischer Chat: wir nutzen das Chat-Modell für den Gesprächsfluss; Recherche/Verifikation rufen wir gezielt separat."""
    sys = build_agent_prompt()
    history = [{"role":"system","content":sys}]
    # Konvertiere bisherige Unterhaltung in Rollen
    for m in st.session_state.messages[-10:]:
        history.append({"role": m["role"], "content": m["content"]})
    history.append({"role":"user","content":question})

    stream = client.responses.stream(
        model=CHAT_MODEL,
        input=history
    )
    buf = []
    with st.chat_message("assistant"):
        ph = st.empty()
        for event in stream:
            if event.type == "response.output_text.delta":
                buf.append(event.delta)
                ph.markdown("".join(buf))
        stream.close()
    text = "".join(buf).strip()
    st.session_state.messages.append({"role":"assistant","content":text})

def finish_pipeline():
    prof = st.session_state.profile
    # 1) web_search candidates
    candidates = run_web_search(client, prof["book"]["title"], prof["book"]["author_guess"])
    # 2) verifier
    verified = pick_best_book(client, prof["book"], candidates)
    st.session_state.verified = verified
    # 3) summary
    summ = summarize_reason(client, prof["book"]["why"], verified.get("title",""), verified.get("author",""))
    st.session_state.summary = summ

    # 4) render SVG
    person_name = user_display or "Member"
    tagline = ""
    svg_text = svg_card(person_name, tagline, prof.get("headline","EXPERT PICKS"), verified, summ)
    st.session_state["last_svg"] = svg_text

    # 5) show preview + downloads
    st.success("Steckbrief fertig!")
    st.download_button("SVG herunterladen", data=svg_text.encode("utf-8"), file_name="steckbrief.svg", mime="image/svg+xml")
    png = rasterize_svg(svg_text)
    st.download_button("PNG herunterladen", data=png, file_name="steckbrief.png", mime="image/png")

# ===== Chat: Eingabe zuerst verarbeiten, dann rendern =====

# 1) Erster Start: Begrüßungsfrage einstellen
if not st.session_state.messages:
    st.session_state.messages = [{"role": "assistant",
                                  "content": "Hi! Lass uns einen Kurz-Steckbrief erstellen. "
                                             "Erste Frage: Welches Buch hat dich beruflich besonders weitergebracht? "
                                             "Bitte den Titel (Autor:in optional)."}]

# 2) Eingabe (zuerst verarbeiten)
user_text = st.chat_input("Antworte dem Bot …")
if user_text:
    # Nutzer-Nachricht anzeigen
    st.session_state.messages.append({"role": "user", "content": user_text})

    # Mini-Interview-State-Machine
    p = st.session_state.profile
    t = user_text.strip()

    bot_msg = None
    if not p["book"]["title"]:
        p["book"]["title"] = t
        bot_msg = "Danke! Warum genau hat dich dieses Buch weitergebracht? 1–2 Sätze."
    elif not p["book"]["why"]:
        p["book"]["why"] = t
        bot_msg = "Super. Kennst du die/den Autor:in? (optional) – oder schreibe „weiter“."
    elif not p["book"]["author_guess"]:
        p["book"]["author_guess"] = t
        bot_msg = "Magst du noch ein hilfreiches Tool nennen (optional)? Name reicht."
    elif not p["tool"]["name"]:
        p["tool"]["name"] = t
        bot_msg = "Kurz: Warum genau dieses Tool? Ein Satz."
    elif not p["tool"]["why"]:
        p["tool"]["why"] = t
        bot_msg = "Optional: Gibt es ein Role Model? (Name) – oder schreibe „fertig“."
    elif not p["role_model"]["name"] and t.lower() != "fertig":
        p["role_model"]["name"] = t
        bot_msg = "Warum dieses Role Model? Ein Satz – oder schreibe „fertig“."
    elif not p["role_model"]["why"] and t.lower() != "fertig":
        p["role_model"]["why"] = t
        bot_msg = "Wenn du bereit bist, schreibe „fertig“ oder klicke den Button „Recherche & Validierung starten“."
    else:
        # Freitext → optional Agent-Antwort (nur wenn API vorhanden)
        if st.session_state.get("agent_ready"):
            ask_agent(t)  # streamt live ins UI
            bot_msg = None  # ask_agent hat schon gerendert
        else:
            bot_msg = "(Demo-Modus ohne OpenAI-Key)"

    if bot_msg:
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})

    # Sofort neu rendern, damit die Antwort sichtbar ist
    st.rerun()

# 3) Jetzt den aktuellen Verlauf rendern
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 4) Finish/Reset Buttons (wie gehabt)
col1, col2 = st.columns(2)
with col1:
    if st.button("🔎 Recherche & Validierung starten"):
        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY fehlt. Lege ihn in den Streamlit-Secrets an.")
        else:
            with st.status("Suche Buchkandidaten, prüfe Quellen, erstelle Zusammenfassung …", expanded=True) as status:
                finish_pipeline()
                status.update(label="Fertig.", state="complete", expanded=False)
with col2:
    if st.button("🧹 Reset"):
        for k in ["messages","profile","verified","summary","last_svg"]:
            if k in st.session_state: del st.session_state[k]
        init_session()
        st.rerun()

