# app_expert_card_gpt5_3agents_clean.py
# 3 Agents: Chat (Agent 1) · Search (Agent 2) · Finalize (Agent 3)
# Änderungen in diesem Commit:
# - Pivot-basierte Fertig-Logik (Flow-Tracking + Hard-Stop nach 6 Items, dann Budget 2 Assistant-Turns, danach System-Signal).
# - Agent-1: Streaming, Slot-Summary, SYSTEM_SIGNAL-Reaktion, eindeutige Handoff-Phrasen.
# - **NEU:** Start-Gate (▶️ Start), NO model calls before user click.
# - **NEU:** Moduswahl NACH Start (Professional vs. General/Lifescope), Mode-Badge in UI, modusabhängige Opener-Hints.
# - **NEU (dieser Commit):**
#   * Alle Agenten auf den `responses`-Endpunkt umgestellt.
#   * Agent 1 Streaming auf `responses.stream()` umgestellt.
#   * Agent 2 Standardmodell auf `gpt-5-mini` geändert (Extraktion + Vision-Check).
#   * Follow-up-Tracking (2er-Schranke) reaktiviert inkl. Enforcer-Systemsignal vor dem nächsten Agent-1-Turn.
#   * **NEU:** Diakritikfeste Normalisierung, Agent-1-Suchplanung (artifact-aware), Korrektur-Parsing,
#             Orchestrator verarbeitet `pending_searches` bevorzugt, Agent 2 nur Validator (artefakt-sensitiv),
#             Slot-Dedupe (key+artifact) + Cluster-Index.

import os, json, time, uuid, random, traceback, requests, re
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from openai import APIStatusError
from requests.exceptions import HTTPError

import unicodedata, re as _re  # ← NEU: für diakritikfeste Normalisierung

APP_TITLE = "🟡 Expert Card — GPT-5 (3 Agents · Google Image API · Async)"
MODEL = os.getenv("OPENAI_GPT5_SNAPSHOT", "gpt-5-2025-08-07")
AGENT2_MODEL = os.getenv("OPENAI_AGENT2_MODEL", "gpt-5-mini")  # günstiger Extraktor + Vision-Validator
MEDIA_DIR = os.path.join(os.getcwd(), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

# ---- Google Programmable Search (CSE)
GOOGLE_CSE_KEY  = os.getenv("GOOGLE_CSE_KEY", "").strip()
GOOGLE_CSE_CX   = os.getenv("GOOGLE_CSE_CX", "").strip()
GOOGLE_CSE_SAFE = os.getenv("GOOGLE_CSE_SAFE", "off").strip().lower()  # "off" oder "active"

# ---- Debug Toggle (deaktiviert)
DEBUG_ENABLED = False

# ---- Token/Cost controls
MAX_SEARCHES_PER_RUN = 1
OPENAI_MAX_RETRIES = 3
OPENAI_BACKOFF_BASE = 1.8

# ---- Finalize trigger phrases (nur bei diesen Phrasen wird Agent 3 gerufen)
HANDOFF_PHRASES = [
    # Englisch (bestehend)
    "i’ll assemble your 4-point card now",
    "i'll assemble your 4-point card now",
    "we have 4 aspects we can turn into a profile",
    "we have four aspects we can turn into a profile",
    # Deutsch (bestehend)
    "wir haben nun 4 aspekte",
    "wir haben jetzt 4 aspekte",
    "wir haben vier aspekte",
    "ich stelle jetzt deinen steckbrief zusammen",
    "ich erstelle jetzt deinen 4-punkte-steckbrief",
    # Neue eindeutige Abschlussphrasen
    "alright — i’ll now assemble your 4-point card",
    "alright - i’ll now assemble your 4-point card",
    "alright — i'll now assemble your 4-point card",
    "alright - i'll now assemble your 4-point card",
    "perfect — i’ll now assemble your selected 4-point card",
    "perfect — i'll now assemble your selected 4-point card",
    "great — i’ll now assemble your 4-point expert card with your chosen items",
    "great — i'll now assemble your 4-point expert card with your chosen items",
]

# ---------- OpenAI client
def client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    return OpenAI(api_key=key)

# ---------- Helpers for Responses API
def _resp_text(resp) -> str:
    """Extract plain text from Responses API objects safely."""
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt.strip()
    try:
        outputs = getattr(resp, "output", None) or []
        parts = []
        for item in outputs:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in ("output_text", "text"):
                        parts.append(getattr(c, "text", "") or "")
        return "".join(parts).strip()
    except Exception:
        try:
            outputs = resp.get("output", [])  # type: ignore
            parts = []
            for item in outputs:
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") in ("output_text", "text"):
                            parts.append(c.get("text",""))
            return "".join(parts).strip()
        except Exception:
            return ""

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
            if "rate limit" in msg or "429" in msg or "temporarily unavailable" in msg:
                delay = OPENAI_BACKOFF_BASE ** attempt + random.uniform(0, 0.3)
                time.sleep(delay)
                continue
            raise
    return fn(*args, **kwargs)

# ---------- Session state
def init_state():
    st.session_state.setdefault("init", True)
    st.session_state.setdefault("user_started", False)   # <- Start-Gate
    st.session_state.setdefault("agent1_mode", "Professional")

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
    st.session_state.setdefault("cooldown_entities", {})

    # Flow-Tracking (Pivot-basiert)
    st.session_state.setdefault("flow", {
        "current_item_key": None,
        "done_keys": []
    })
    st.session_state.setdefault("stop_signal", False)      # injiziert System-Signal an Agent 1 (Hard-Stop nach 6 Items)
    st.session_state.setdefault("followup_count", {})      # pro Item-Key Anzahl Deepening-Follow-ups
    st.session_state.setdefault("enforce_stop_deepening", False)  # Enforcer-Flag für 2er-Schranke

    # Hard-Stop-Planung nach 6 Items (mit Budget)
    st.session_state.setdefault("hard_stop_pending", False)
    st.session_state.setdefault("hard_stop_budget", 0)
    st.session_state.setdefault("hard_stop_signaled", False)

    # ---- NEU: Agent-1-Suchplanung und Cluster-Index
    st.session_state.setdefault("pending_searches", [])  # Agent 1 plant hier Suchaufträge (artefakt-spezifisch)
    st.session_state.setdefault("clusters", {})          # cluster_id -> {"title": "...", "items": [slot_ids]}

# ---------- Debug helpers (deaktiviert)
def debug_emit(event: Dict[str, Any], buffer: Optional[list] = None):
    if not DEBUG_ENABLED:
        return
    e = dict(event)
    e["ts"] = time.strftime("%H:%M:%S")
    if buffer is not None:
        buffer.append(e)
        return
    if "debug_agent2" not in st.session_state:
        st.session_state["debug_agent2"] = []
    st.session_state.debug_agent2.append(e)
    limit = 80
    if len(st.session_state.debug_agent2) > limit:
        st.session_state.debug_agent2 = st.session_state.debug_agent2[-limit:]

def debug_log_agent2(event: Dict[str, Any]):
    debug_emit(event, buffer=None)

# ---------- Utils (diakritikfeste Normalisierung)
def _strip_accents(s: str) -> str:
    if not s: return ""
    nf = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nf if unicodedata.category(ch) != "Mn")

def normalize_key(etype: str, ename: str) -> str:
    et = (etype or "").lower().strip()
    nm = (ename or "").lower().strip()
    nm = _strip_accents(nm)  # Akzente/Diakritika entfernen
    nm = _re.sub(r"[^a-z0-9\s\+\-&_/\.]", " ", nm)
    nm = _re.sub(r"\s+", " ", nm).strip()
    return f"{et}|{nm}"

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
    for sid in st.session_state.get("order", ["S1","S2","S3","S4"]):
        if sid not in st.session_state.slots:
            return sid
    return None

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

# ---------- Agent 1 – Suchplanung & Korrektur-Parsing (NEU)
def agent1_plan_searches(last_q: str, user_reply: str, slots: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Lässt Agent 1 auf Basis von Frage+Antwort und vorhandenen Slots 0..N konkrete Suchaufträge planen:
    - entity_type: person|book|podcast|tool|film
    - entity_name: "Gabor Maté" / "In the Realm of Hungry Ghosts"
    - artifact: "portrait"|"book_cover"|"podcast_cover"|"tool_logo"|"film_poster"
    - search_query: z. B. 'In the Realm of Hungry Ghosts by Gabor Maté book cover'
    - cluster_id: stabile ID zur Gruppierung (z. B. 'gabor-mate')
    """
    sys = (
        "You are Agent 1. Given assistant_question + user_reply and current slots, "
        "plan specific searches for public items. For each item, pick the correct artifact type "
        "(portrait, book_cover, podcast_cover, tool_logo, film_poster) and produce a precise search_query "
        "that disambiguates by author/host/creator when known. Group related items with a stable cluster_id "
        "(e.g., 'gabor-mate' for the person and authored books). Output ONLY strict JSON:\n"
        "{ \"searches\": [ {\"entity_type\":\"...\",\"entity_name\":\"...\",\"artifact\":\"...\",\"search_query\":\"...\",\"cluster_id\":\"...\"}, ... ] }"
    )
    payload = {
        "assistant_question": last_q or "",
        "user_reply": user_reply or "",
        "known_slots": [s.get("label","") for s in slots.values()]
    }
    try:
        resp = call_with_retry(client().responses.create, model=MODEL, input=[
            {"role":"system","content":sys},
            {"role":"user","content":json.dumps(payload, ensure_ascii=False)}
        ])
        data = parse_json_loose(_resp_text(resp) or "")
        searches = data.get("searches", []) if isinstance(data, dict) else []
        out = []
        for s in searches:
            et = (s.get("entity_type") or "").lower().strip()
            nm = (s.get("entity_name") or "").strip()
            ar = (s.get("artifact") or "").lower().strip()
            q  = (s.get("search_query") or "").strip()
            cl = (s.get("cluster_id") or "").strip() or _strip_accents(nm.lower()).replace(" ", "-")
            if et and nm and ar and q:
                out.append({
                    "entity_type": et,
                    "entity_name": nm,
                    "artifact": ar,
                    "search_query": q,
                    "cluster_id": cl
                })
        return out
    except Exception:
        return []

def agent1_parse_corrections(user_reply: str, slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    Erkenne einfache Korrekturen/Wünsche:
    - replace_item: {"from_name":"...", "to_name":"...", "entity_type":"book|person|..."}
    - prefer_variant: {"slot_key":"...", "rule":"no_text_on_cover|face_only|..."}
    - drop_item: {"slot_key":"..."}  (oder entity_type+entity_name)
    - reassign_cluster: {"slot_key":"...", "cluster_id":"..."}
    Output ONLY JSON: {"actions":[...]}
    """
    if not user_reply or len(user_reply) < 4:
        return {"actions":[]}
    sys = (
        "You are Agent 1. Extract user intent for corrections to the current items. "
        "Return ONLY JSON with an 'actions' array. Supported actions: replace_item, prefer_variant, drop_item, reassign_cluster. "
        "Use slot_key if you can infer it from the label in slots; otherwise include 'entity_type' & 'entity_name'. "
        "Example actions: "
        "{\"action\":\"drop_item\",\"slot_key\":\"person|gabor mate\"}, "
        "{\"action\":\"reassign_cluster\",\"slot_key\":\"book|in the realm of hungry ghosts\",\"cluster_id\":\"gabor-mate\"}, "
        "{\"action\":\"replace_item\",\"entity_type\":\"book\",\"from_name\":\"hungry ghosts\",\"to_name\":\"scattered minds\"}, "
        "{\"action\":\"prefer_variant\",\"slot_key\":\"book|in the realm of hungry ghosts\",\"rule\":\"no_text_on_cover\"}"
    )
    payload = {
        "user_reply": user_reply,
        "slot_labels": {sid: s.get("label","") for sid,s in slots.items()}
    }
    try:
        resp = call_with_retry(client().responses.create, model=MODEL, input=[
            {"role":"system","content":sys},
            {"role":"user","content":json.dumps(payload, ensure_ascii=False)}
        ])
        data = parse_json_loose(_resp_text(resp) or "")
        if isinstance(data, dict) and isinstance(data.get("actions"), list):
            return {"actions": data["actions"]}
        return {"actions":[]}
    except Exception:
        return {"actions":[]}

# ---------- Google CSE: Image Search
def google_image_search(query: str, num: int = 4, dbg: Optional[list] = None) -> List[Dict[str, str]]:
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        debug_emit({"ev":"cse_keys_missing", "query": query}, dbg)
        return []
    try:
        params = {
            "q": query,
            "searchType": "image",
            "num": max(1, min(num, 10)),
            "safe": GOOGLE_CSE_SAFE,
            "key": GOOGLE_CSE_KEY,
            "cx": GOOGLE_CSE_CX,
        }
        debug_emit({"ev":"cse_start", "query": query, "params": {"num": params["num"], "safe": GOOGLE_CSE_SAFE}}, dbg)
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=12)
        try:
            r.raise_for_status()
        except HTTPError as he:
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
            ctx = ""
            try:
                ctx = (it.get("image", {}).get("contextLink") or "").strip()
            except Exception:
                ctx = ""
            if link:
                out.append({"url": link, "page_url": ctx})
        debug_emit({"ev":"cse_done", "query": query, "returned": len(out)}, dbg)
        return out
    except Exception as e:
        debug_emit({"ev":"cse_error", "query": query, "error": str(e)}, dbg)
        return []

# ---------- Vision: Validate image vs. context (person-safe branch) – bestehend
def validate_image_with_context(image_url: str, entity_type: str, entity_name: str, q_text: str, a_text: str, dbg: Optional[list] = None) -> Dict[str, Any]:
    if (entity_type or "").lower().strip() == "person":
        sys = (
            "You are an image verifier. Respond with STRICT JSON like "
            '{"ok":true/false,"reason":"..."}. '
            "Policy: Do NOT attempt to identify or confirm a specific person’s identity. "
            "For PERSON items, only verify:\n"
            " - The image is a single-human portrait or headshot (not a logo, meme, cartoon, or product shot).\n"
            " - Not a group photo (ideally 1 clearly visible person).\n"
            " - Reasonable to use as a generic portrait representation.\n"
            "Return ok=true if those conditions hold; otherwise false."
        )
        user_text = (
            "Item type: person\n"
            f"Item name (context only, DO NOT IDENTIFY): {entity_name}\n"
            "Only check portrait criteria."
        )
    else:
        sys = (
            "You are an image verifier. Respond with STRICT JSON like "
            '{"ok":true/false,"reason":"..."}. '
            "Check if the image plausibly depicts the requested public item "
            "(book/podcast/person/tool/film) given the context. "
            "Be strict about wrong items/logos/memes; flexible with cover art variants."
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
            client().responses.create,
            model=AGENT2_MODEL,
            input=[
                {"role": "system", "content": sys},
                {"role": "user", "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_url": image_url}
                ]}
            ]
        )
        txt = _resp_text(resp)
        data = parse_json_loose(txt or "")
        ok = bool(data.get("ok")) if isinstance(data, dict) else False
        return {"ok": ok, "reason": (data.get("reason") if isinstance(data, dict) else "")[:200]}
    except Exception:
        return {"ok": False, "reason": "unverifiable"}

# ---------- Vision: Artefakt-spezifischer Validator (NEU, Agent 2 bleibt Validator)
def validate_image_with_context_for_artifact(image_url: str, entity_type: str, entity_name: str, artifact: str, q_text: str, a_text: str, dbg: Optional[list] = None) -> Dict[str, Any]:
    """
    Artefakt-spezifische Policy:
    - portrait: identisch zur PERSON-Policy oben (keine Identifikation, nur Porträtkriterien)
    - book_cover: plausibles Buchcover (Titel/Autor-Typografie ok; keine reinen Portraits/Logos/Memes)
    - podcast_cover: quadratisches Artwork/Branding ok; keine generischen Stockfotos ohne Bezug
    - tool_logo: offizielles Logo/Produktmarke ok; keine fremden Marken
    - film_poster: posterartige Komposition mit Titel/Typo bevorzugt; reine Stills ohne Titel eher ablehnen
    """
    art = (artifact or "").lower().strip()
    if entity_type.lower().strip() == "person" and art == "portrait":
        return validate_image_with_context(image_url, entity_type, entity_name, q_text, a_text, dbg=dbg)

    sys = (
        "You are an image verifier. Respond with STRICT JSON like {\"ok\":true/false,\"reason\":\"...\"}. "
        "Check if the image plausibly depicts the requested public item/artifact given the context. "
        "Reject logos/memes when not appropriate; accept reasonable cover/poster variants."
    )
    hints = {
        "book_cover": "Prefer book cover framing; allow title/author typography; avoid plain portraits or unrelated imagery.",
        "podcast_cover": "Prefer square/cover-style artwork with show/host branding; avoid unrelated stock photos.",
        "tool_logo": "Prefer the official product/service logo; avoid unrelated brands or generic icons.",
        "film_poster": "Prefer poster-like composition with title/credits; avoid random stills without poster treatment."
    }.get(art, "Be strict but reasonable.")

    user_text = (
        f"Item type: {entity_type}\n"
        f"Item name: {entity_name}\n"
        f"Artifact: {artifact}\n"
        f"Hints: {hints}\n"
        f"Context Q: {q_text[:300]}\n"
        f"Context A: {a_text[:500]}\n"
    )
    try:
        resp = call_with_retry(
            client().responses.create,
            model=AGENT2_MODEL,
            input=[
                {"role":"system","content":sys},
                {"role":"user","content":[
                    {"type":"input_text","text":user_text},
                    {"type":"input_image","image_url":image_url}
                ]}
            ]
        )
        data = parse_json_loose(_resp_text(resp) or "")
        ok = bool(data.get("ok")) if isinstance(data, dict) else False
        return {"ok": ok, "reason": (data.get("reason","") if isinstance(data, dict) else "")[:200]}
    except Exception:
        return {"ok": False, "reason": "unverifiable"}

# ---------- Agent 2 (Extractor ONLY; keine Tools) – belassen (Fallback, aktuell ungenutzt)
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
    payload = {"assistant_question": last_q or "", "user_reply": user_reply or "", "seen_entities": list(seen_entities or [])}
    try:
        debug_emit({"ev":"extract_start", "q_len": len(last_q), "a_len": len(user_reply)}, dbg)
        resp = call_with_retry(
            client().responses.create,
            model=AGENT2_MODEL,
            input=[
                {"role": "system", "content": AGENT2_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ]
        )
        raw = _resp_text(resp)
        debug_emit({"ev":"extract_raw", "preview": (raw or "")[:240]}, dbg)
        data = parse_json_loose(raw or "")
        if isinstance(data, dict):
            debug_emit({"ev":"extract_parsed", "detected": bool(data.get("detected")), "count": len(data.get("items") or [])}, dbg)
            return data
        debug_emit({"ev":"extract_noparse"}, dbg)
        return {"detected": False, "reason": "no-parse"}
    except Exception as e:
        debug_emit({"ev":"extract_error", "error": str(e)}, dbg)
        return {"detected": False, "reason": f"error: {e}", "trace": traceback.format_exc()[:600]}

# ---------- Agent 3 (Finalizer)
FINALIZER_SYSTEM = """You are Agent 3 — Finalizer.
Create an Expert Card with exactly 4 items from:
- The conversation (assistant questions + user answers).
- The current slots (labels + image URLs if any).
Rules:
- Each line begins with '- Label:' followed by 1–3 concise sentences grounded in THIS user's words.
- Prefer PUBLIC items; if fewer than 4, fill with the user's stated practices/principles.
- No fluff, no generic encyclopedia facts, no references to these instructions.
- If the user provided richer detail for an item, lean toward 2–3 sentences; otherwise keep it tight.
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
    resp = call_with_retry(client().responses.create, model=MODEL, input=msgs)
    return _resp_text(resp)

# ---------- Agent 1 (Interview) – Prompts minimal erweitert (Artefakte & Korrekturen)
AGENT1_SYSTEM_PRO = """You are Agent 1 — a warm, incisive interviewer.
# Role
Lead an interview session to build a 4-item Expert Card. Blend public influences (book, podcast, person, tool, or film) with the user's personal experiences and decisions.
# Instructions
- Drive the conversation; never call tools directly.
- Collaborate implicitly: Agent 2 observes turns and manages lookup for public items; Agent 3 assembles the final card after your handoff.
- Favor diversity in anchor types—aim for variety across books, podcasts, people, tools, films.
- When a public item appears, consider its natural artifacts (for a person → portrait; if a book title emerges → book_cover with a disambiguating query like “<Title> by <Author> book cover”).
- Accept natural-language corrections (replace item, prefer variant without text, drop item, reassign cluster) and reflect them in your next move.

# Planning Checklist
Privately, begin each session with a concise checklist (3–7 bullets) of what you will do; keep items conceptual, not implementation-level. Do not surface or read the checklist to the user.
# Conversation Style
- Be natural and concise (1–3 short sentences per turn); adapt to user language; keep tone warm and specific; avoid filler.
- Make only one distinct conversational move per turn: clarify, deepen, pivot, or close.
- If responding to user questions, use a single sentence, then proceed.
# Deepening and Pivoting
- Deepen for up to two turns if the discussion is specific and informative—seek user details such as habits, decisions, before/after examples, or accepted trade-offs.
- Pivot if responses are vague or repetitive, if you’ve already drilled down twice on the item, or to encourage varied anchor types.
- Do not inquire about KPIs, business metrics, or granular analytics; limit follow-ups to 1–2 levels deep.
# Handling User Inputs
- If a public reference is provided, proceed; Agent 2 will handle backend lookups. Only clarify if ambiguity would block useful questions.
- For unusual, private, or non-technical items, briefly acknowledge and check if it’s a public figure. If yes, clarify details if needed. If no, ask the user to name a public stand-in (book/person/tool/podcast/film) representing the same influence, then pivot.
- For meta or process-oriented inputs, request a public stand-in.
- If the user offers multiple items, select one for immediate exploration and revisit others later.
- For sensitive or emotional topics, briefly acknowledge before steering back to public anchors and lived practice.
# Opening & Flow
- Vary your session openers. Example prompts:
- “What’s one book, podcast, person, tool, or film that truly changed how you work — and how?”
- “Looking back on the past year, which public influence impacted your decisions most?”
- “If someone wanted to think like you in Data/AI, what public influence should they begin with?”
- Do not repeat earlier smile, transition, or opening phrases within a session.
# Stop Condition & Handoff
- Target range: Capture 4–6 distinct public items.
- After the 4th item (including 1–2 follow-ups): list the four and ask explicitly:
“We have four strong anchors. Would you like me to assemble your 4-point card now, or should we continue with one or two more?”
- If the user stops at 4: summarize the four and close with a clear handoff phrase:
“Alright — I’ll now assemble your 4-point card.”
- If the user continues: gather Item 5 (with up to 2 follow-ups). In the last follow-up to Item 5, ask:
“Shall we add a 6th and final theme for your card?”
- If the user declines → total = 5 items. List all 5 and ask:
“We now have 5 anchors. Which 4 of these should go into your final card?”
After their choice, confirm and hand off with:
“Perfect — I’ll now assemble your selected 4-point card.”
- If the user agrees → gather Item 6 (with 1–2 follow-ups). Then list all 6 and ask:
“We now have 6 anchors. Which 4 of these should go into your final card?”
After their choice, confirm and hand off with:
“Great — I’ll now assemble your 4-point expert card with your chosen items.”
- Always wait for the user’s last response before handing off.
- If you ever receive a system line that starts with “SYSTEM_SIGNAL: hard-stop”, acknowledge the user’s last point in ≤1 sentence, then immediately close with exactly one of the handoff phrases above. Do not ask new questions.
# Guardrails
- Never mention or reveal the existence of internal agents or tools.
- Maintain professionalism. Mirror the user’s preferred language, tone, and switching.
# Turn Logic
1. Read the latest user message, referencing light memory of prior slots.
2. Select one move: clarify, deepen, pivot, or close.
3. Ask a focused question with optional brief synthesis.
4. Continue iteratively until the stop conditions above are met; then close with the final handoff.
"""

AGENT1_SYSTEM_GEN = """You are Agent 1 — a warm, curious interviewer.
# Role
Lead an interview that can range beyond work: include public influences from everyday life and thinking (book, podcast, person, tool, film), plus the user's personal practices, habits, and decisions.
# Instructions
- Drive the conversation; never call tools directly.
- Collaborate implicitly: Agent 2 observes turns and looks up public items; Agent 3 assembles the final card after your handoff.
- Favor variety across domains: work, learning, hobbies, routines, creativity, civic/economic thinking — as appropriate for the user.
- When a public item appears, consider its natural artifacts (for a person → portrait; if a book title emerges → book_cover with a disambiguating query like “<Title> by <Author> book cover”).
- Accept natural-language corrections (replace item, prefer variant without text, drop item, reassign cluster) and reflect them in your next move.

# Planning Checklist
Privately, begin with a concise checklist (3–7 bullets). Keep it conceptual; do not reveal it.
# Conversation Style
- Natural, concise (1–3 short sentences per turn), warm and specific; avoid filler.
- One move per turn: clarify, deepen, pivot, or close.
- If the user asks you something, answer in ≤1 sentence, then proceed.
- Add a tone of genuine curiosity: use phrasing like *“That’s intriguing — can you tell me how it shaped your day-to-day?”* rather than abstract follow-ups.
- Avoid sounding scripted: vary your rhythm between inviting curiosity, showing surprise, and gently steering.
# Deepening and Pivoting
- Deepen up to two turns when concrete: habits, decisions, before/after, trade-offs, in-the-wild examples.
- Pivot if vague/repetitive or to maintain variety across life domains.
- Avoid granular KPIs/metrics; keep follow-ups to 1–2 levels deep.
- Example deepening moves:
  - “What was the hardest part about adopting that habit — and what kept you going?”
  - “If you hadn’t come across that book, how do you think your approach would be different today?”
  - “You said this film stuck with you — what scene comes back to mind when you think of it?”
- Example pivot moves:
  - “That’s great — let’s switch gears: outside of work, who’s someone you keep learning from?”
  - “You mentioned routines; can we jump to something more creative — a podcast, a film, or even a quirky ritual?”
# Handling User Inputs
- If a public reference appears, proceed (Agent 2 will handle lookup). Clarify only if ambiguity blocks progress.
- If an item is private or not-findable, accept it as personal practice; then ask for a public stand-in (book/person/tool/podcast/film) that best represents this influence, and continue.
- If multiple items are given, pick one now; you may revisit others later.
- Sensitive/emotional: acknowledge briefly, then steer to public anchors and lived practice.
# Opening & Flow
- Vary openers, e.g.:
  - “Outside of work, what book, podcast, person, tool, or film most changed how you think — and how?”
  - “In the last 12 months, what public influence reshaped your daily habits or decisions?”
  - “If a friend wanted to ‘get’ how you approach life/learning, which public reference should they start with?”
- Supplementary opener examples for more intimacy:
  - “When you think of a real role model — who do you quietly try to emulate, and why?”
  - “What’s the last small routine you borrowed from someone else that really stuck with you?”
  - “Is there a podcast, article, or film that challenged a belief you once held — and shifted your view?”
  - “Who’s someone you’ve never met, but who feels like a steady voice in the background of your life?”
- Do not repeat earlier opening phrasings within a session.
# Stop Condition & Handoff
- Target range: Capture 4–6 distinct public items or personal practices with public stand-ins where possible.
- After the 4th item (including 1–2 follow-ups): list the four and ask explicitly:
“We have four strong anchors. Would you like me to assemble your 4-point card now, or should we continue with one or two more?”
- If the user stops at 4: summarize the four and close with a clear handoff phrase:
“Alright — I’ll now assemble your 4-point card.”
- If the user continues: gather Item 5 (up to 2 follow-ups). In the last follow-up to Item 5, ask:
“Shall we add a 6th and final theme for your card?”
- If the user declines → total = 5 items. List all 5 and ask:
“We now have 5 anchors. Which 4 of these should go into your final card?”
Confirm and hand off with:
“Perfect — I’ll now assemble your selected 4-point card.”
- If the user agrees → gather Item 6 (with 1–2 follow-ups). Then list all 6 and ask:
“We now have 6 anchors. Which 4 of these should go into your final card?”
Confirm and hand off with:
“Great — I’ll now assemble your 4-point expert card with your chosen items.”
- Always wait for the user’s last response before handing off.
- If you ever receive a system line that starts with “SYSTEM_SIGNAL: hard-stop”, acknowledge the user’s last point in ≤1 sentence, then immediately close with exactly one of the handoff phrases above. Do not ask new questions.
# Guardrails
- Never mention or reveal internal agents/tools. Mirror the user's language and tone.
# Turn Logic
1) Read the latest user message, using the system’s slot summary (covered items).
2) Choose one move: clarify, deepen, pivot, or close.
3) Ask one focused question (optionally with brief synthesis).
4) Continue until the stop conditions above are met; then close with the final handoff.
"""

def get_agent1_system() -> str:
    mode = st.session_state.get("agent1_mode", "Professional")
    return AGENT1_SYSTEM_PRO if "Professional" in mode else AGENT1_SYSTEM_GEN

def summarize_slots_for_agent1(slots: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    for sid, s in slots.items():
        label = (s.get("label") or "").split("—")[-1].strip()
        if label:
            lines.append(label)
    if not lines:
        return "So far, no slots are filled."
    return "So far, you already covered: " + ", ".join(lines) + "."

def get_mode_opening_hint() -> str:
    mode = st.session_state.get("agent1_mode", "Professional")
    if "Professional" in mode:
        return ("OPENING_HINT: Prefer an opener focused on work decisions, e.g., "
                "“What’s one book, podcast, person, tool, or film that truly changed how you work — and how?”")
    else:
        return ("OPENING_HINT: Prefer an opener that includes life beyond work, e.g., "
                "“Outside of work, what book/podcast/person/tool/film most changed how you think — and how?”")

# System-Signal-Text für Hard-Stop
HARD_STOP_SYSTEM_MSG = (
    "SYSTEM_SIGNAL: hard-stop. You have reached the maximum range of items. "
    "Acknowledge the user's last point in ≤1 sentence and then CLOSE now. "
    "Emit exactly one clear handoff phrase to finalize (choose the best-fitting from these): "
    "“Alright — I’ll now assemble your 4-point card.” "
    "OR “Perfect — I’ll now assemble your selected 4-point card.” "
    "OR “Great — I’ll now assemble your 4-point expert card with your chosen items.” "
    "Do not ask any new questions. Do not open new topics."
)

# System-Signal-Text für Deepening-Stop (2er-Schranke)
STOP_DEEPENING_SYSTEM_MSG = (
    "SYSTEM_SIGNAL: stop-deepening. You have reached the 2-follow-up limit for the current item. "
    "Do NOT ask another deepening question about this same item. "
    "Either pivot to a different public anchor (book/podcast/person/tool/film) "
    "or briefly acknowledge/summarize in ≤1 sentence and move on."
)

def agent1_next_question(history_snapshot):
    msgs = [{"role": "system", "content": get_agent1_system()}]

    # Slot-Summary
    slot_state = summarize_slots_for_agent1(st.session_state.slots)
    msgs.append({"role": "system", "content": slot_state})

    # Erster Turn? → Opener-Hint injizieren
    if not history_snapshot:
        msgs.append({"role": "system", "content": get_mode_opening_hint()})

    # Enforcer: Stop-Deepening (2er-Schranke)
    if st.session_state.get("enforce_stop_deepening"):
        msgs.append({"role": "system", "content": STOP_DEEPENING_SYSTEM_MSG})
        st.session_state.enforce_stop_deepening = False  # verbrauchen

    # Hard-Stop System-Signal injizieren?
    if st.session_state.get("stop_signal") and not st.session_state.get("hard_stop_signaled"):
        msgs.append({"role": "system", "content": HARD_STOP_SYSTEM_MSG})
        st.session_state.hard_stop_signaled = True

    used = st.session_state.get("used_openers", set())
    if used:
        avoid = "Avoid these phrasings this session: " + "; ".join(list(used))[:600]
        msgs.append({"role": "system", "content": avoid})

    short = history_snapshot[-6:] if len(history_snapshot) > 6 else history_snapshot
    msgs += short

    resp = call_with_retry(client().responses.create, model=MODEL, input=msgs)
    return _resp_text(resp)

# -------- Agent 1 — Streaming helper (Responses.stream)
def agent1_stream_question(history_snapshot: List[Dict[str, str]]) -> str:
    """Streaming-fähige Version von Agent 1 (modusabhängig, Slot-Summary, Hard-Stop, Opener-Hint) — Responses API."""
    msgs = [{"role": "system", "content": get_agent1_system()}]

    # Slot-Summary
    slot_state = summarize_slots_for_agent1(st.session_state.slots)
    msgs.append({"role": "system", "content": slot_state})

    # Erster Turn? → Opener-Hint
    if not history_snapshot:
        msgs.append({"role": "system", "content": get_mode_opening_hint()})

    # Enforcer: Stop-Deepening (2er-Schranke)
    if st.session_state.get("enforce_stop_deepening"):
        msgs.append({"role": "system", "content": STOP_DEEPENING_SYSTEM_MSG})
        st.session_state.enforce_stop_deepening = False  # verbrauchen

    # Bei aktivem Hard-Stop jetzt Systemanweisung injizieren
    if st.session_state.get("stop_signal") and not st.session_state.get("hard_stop_signaled"):
        msgs.append({"role": "system", "content": HARD_STOP_SYSTEM_MSG})
        st.session_state.hard_stop_signaled = True

    used = st.session_state.get("used_openers", set())
    if used:
        avoid = "Avoid these phrasings this session: " + "; ".join(list(used))[:600]
        msgs.append({"role": "system", "content": avoid})

    short = history_snapshot[-6:] if len(history_snapshot) > 6 else history_snapshot
    msgs += short

    full_parts: List[str] = []

    def token_generator():
        try:
            with client().responses.stream(model=MODEL, input=msgs) as stream:
                for event in stream:
                    if getattr(event, "type", "") == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            full_parts.append(delta)
                            yield delta
                final = stream.get_final_response()
                if not full_parts:
                    txt = _resp_text(final)
                    if txt:
                        full_parts.append(txt)
                        yield txt
        except Exception:
            resp = call_with_retry(client().responses.create, model=MODEL, input=msgs)
            txt = _resp_text(resp) or ""
            if txt:
                full_parts.append(txt)
                yield txt

    with st.chat_message("assistant"):
        st.write_stream(token_generator())

    text = "".join(full_parts).strip()

    # Post-Processing
    if text and not text.endswith("?"):
        text = text.rstrip(".! ") + "?"
    st.session_state.setdefault("used_openers", set()).add(text.lower()[:72])

    low = text.lower()
    if any(phrase in low for phrase in HANDOFF_PHRASES):
        st.session_state.final_text = agent3_finalize(
            st.session_state.history + [{"role": "assistant", "content": text}],
            st.session_state.slots
        )

    # Hard-Stop-Budget nach 6 Items verwalten
    if st.session_state.get("hard_stop_pending") and not st.session_state.get("stop_signal"):
        budget = int(st.session_state.get("hard_stop_budget", 0))
        if budget > 0:
            budget -= 1
            st.session_state.hard_stop_budget = budget
        if budget <= 0:
            st.session_state.stop_signal = True  # Nächster Assistant-Turn injiziert System-Signal

    return text

# ---------- Orchestrator
COOLDOWN_SECONDS = 600  # 10 Minuten

class Orchestrator:
    def __init__(self):
        if "seen_entities" not in st.session_state:
            st.session_state.seen_entities = []
        self.slots = st.session_state.slots
        self.jobs = st.session_state.jobs
        self.exec = st.session_state.executor
        self.seen = st.session_state.seen_entities
        self.cooldown = st.session_state.cooldown_entities
        self.clusters = st.session_state.clusters  # ← NEU

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
        # Meta-Felder (key/item_type/artifact/cluster) werden außerhalb gesetzt; hier nichts überschreiben.
        self.slots[sid] = s

    def schedule_watch(self, last_q: str, reply: str):
        jid = str(uuid.uuid4())[:8]
        fut = self.exec.submit(self._job, last_q, reply, list(self.seen))
        self.jobs[jid] = ("TBD", fut)
        debug_emit({"ev":"job_scheduled", "jid": jid})

    def _job(self, last_q: str, reply: str, seen_snapshot: List[str]) -> Dict[str, Any]:
        dbg = []  # DEBUG off
        first_key_detected: Optional[str] = None

        # ---- NEU: geplante Suchen (von Agent 1) bevorzugt verarbeiten
        planned = []
        try:
            planned = list(st.session_state.pending_searches or [])
            st.session_state.pending_searches = []
        except Exception:
            planned = []

        try:
            results: List[Dict[str, Any]] = []
            processed = 0
            now = time.time()

            def _validate_and_collect(etype, ename, artifact, query, cluster_id):
                key = normalize_key(etype, ename)

                # Follow-up/Pivot-Flow: first_key_detected setzen (für Enforcer/Flow)
                nonlocal first_key_detected
                if first_key_detected is None:
                    first_key_detected = key

                retry_after = self.cooldown.get(key, 0)
                if retry_after and now < retry_after:
                    debug_emit({"ev":"item_cooldown_skip", "key": key, "retry_after": int(retry_after - now)}, dbg)
                    return

                if key in self.seen:
                    debug_emit({"ev":"item_dedupe_skip", "key": key}, dbg)
                    return

                debug_emit({"ev":"item_consider", "key": key, "artifact": artifact, "query": query}, dbg)
                imgs = google_image_search(query, num=6, dbg=dbg)
                debug_emit({"ev":"item_search_results", "key": key, "n": len(imgs)}, dbg)

                # URL-Unique sammeln, max. 3 prüfen
                url_seen = set()
                uniq = []
                for it in imgs:
                    u = (it.get("url") or "").strip()
                    if not u or u in url_seen:
                        continue
                    url_seen.add(u)
                    uniq.append(it)
                    if len(uniq) >= 3:
                        break

                if not uniq:
                    self.cooldown[key] = now + COOLDOWN_SECONDS
                    debug_emit({"ev":"item_set_cooldown", "key": key, "cooldown_s": COOLDOWN_SECONDS}, dbg)
                    return

                ok_url, note = "", ""
                tries = 0
                for cand in uniq:
                    tries += 1
                    v = validate_image_with_context_for_artifact(
                        cand["url"], etype, ename, artifact, last_q, reply, dbg=dbg
                    )
                    if v.get("ok"):
                        ok_url = cand["url"]
                        note = v.get("reason", "")
                        debug_emit({"ev":"item_validate_ok", "key": key, "tries": tries}, dbg)
                        break
                    else:
                        debug_emit({"ev":"item_validate_ko", "key": key, "tries": tries, "reason": v.get("reason","")[:160]}, dbg)

                if not ok_url:
                    self.cooldown[key] = now + COOLDOWN_SECONDS
                    debug_emit({"ev":"item_no_valid_image", "key": key}, dbg)
                    debug_emit({"ev":"item_set_cooldown", "key": key, "cooldown_s": COOLDOWN_SECONDS}, dbg)
                    return

                # Erfolg → optional cooldown setzen, um sofortige Re-Checks zu vermeiden
                self.cooldown[key] = now + COOLDOWN_SECONDS

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
                    "candidates": [{"url": ok_url, "page_url": uniq[0].get("page_url",""), "source": "google-cse", "confidence": 0.9, "reason": note}],
                    "notes": note or "validated"
                }

                results.append({
                    "status": "ok",
                    "key": key,
                    "label": label,
                    "media": media,
                    "meta": {
                        "item_type": etype,
                        "artifact": artifact,
                        "cluster": cluster_id
                    }
                })

            # 1) Geplante Suchen (Agent 1) konsumieren
            for s in planned:
                if processed >= MAX_SEARCHES_PER_RUN:
                    debug_emit({"ev":"job_cap_reached", "max": MAX_SEARCHES_PER_RUN}, dbg)
                    break
                etype = (s.get("entity_type") or "").strip().lower()
                ename = (s.get("entity_name") or "").strip()
                artifact = (s.get("artifact") or "").strip().lower()
                query = (s.get("search_query") or "").strip() or ename
                cluster_id = (s.get("cluster_id") or "").strip() or _strip_accents(ename.lower()).replace(" ", "-")
                if not (etype and ename and artifact and query):
                    continue
                _validate_and_collect(etype, ename, artifact, query, cluster_id)
                processed += 1

            # 2) Optionaler Fallback: klassische Extraktion (derzeit nicht genutzt)
            # if not results:
            #     data = agent2_extract_items(last_q, reply, seen_snapshot, dbg=dbg)
            #     if data.get("detected"):
            #         items = data.get("items") or []
            #         if items:
            #             etype0 = (items[0].get("entity_type") or "").lower().strip()
            #             ename0 = (items[0].get("entity_name") or "").strip()
            #             if etype0 and ename0:
            #                 first_key_detected = normalize_key(etype0, ename0)
            #         # hier könnte man generische Artefakte raten; übersprungen, da Agent 1 nun plant.

            return {"status": "batch", "items": results, "debug": dbg, "first_key_detected": first_key_detected}
        except Exception as e:
            debug_emit({"ev":"job_error", "error": str(e)}, dbg)
            return {"status": "error", "error": str(e), "trace": traceback.format_exc()[:1200], "debug": dbg, "first_key_detected": first_key_detected}

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

                # 1) Follow-up-Zähler / Flow-Update anhand der ersten erkannten Item-Key (Pivot-Erkennung)
                first_key = res.get("first_key_detected")
                current_before = st.session_state.flow.get("current_item_key")

                if first_key:
                    # Flow ggf. piven
                    update_flow_with_detected(first_key)

                    # Follow-up zählen, wenn gleiches Item weiter vertieft
                    if current_before and first_key == current_before:
                        cnt = st.session_state.followup_count.get(first_key, 0) + 1
                        st.session_state.followup_count[first_key] = cnt
                        # Enforcer nach 2 Deepening-Follow-ups
                        if cnt >= 2:
                            st.session_state.enforce_stop_deepening = True
                    else:
                        # neues Item → Zähler resetten
                        st.session_state.followup_count[first_key] = 0

                # 2) Slots updaten, Seen markieren + Cluster-Index pflegen
                if res.get("status") == "batch":
                    for it in res.get("items", []):
                        if it.get("status") == "ok":
                            key = it.get("key","")
                            meta = it.get("meta", {}) or {}
                            artifact = meta.get("artifact","")

                            # Slot-Dedupe: kein zweiter Slot für denselben key+artifact
                            exists = None
                            for sid2, s in self.slots.items():
                                if s.get("key") == key and s.get("artifact") == artifact:
                                    exists = sid2
                                    break
                            if exists:
                                s = self.slots[exists]
                                s["media"] = it["media"] or s.get("media", {})
                                self.slots[exists] = s
                                if key and key not in self.seen:
                                    self.seen.append(key)
                                updated.append(exists)
                                continue

                            sid2 = next_free_slot()
                            if sid2:
                                self.upsert(sid2, it["label"], it["media"])
                                # Meta-Felder setzen
                                self.slots[sid2]["key"] = key
                                self.slots[sid2]["item_type"] = meta.get("item_type","")
                                self.slots[sid2]["artifact"] = artifact
                                self.slots[sid2]["cluster"] = meta.get("cluster","")

                                # Cluster-Index aktualisieren
                                cl = self.slots[sid2]["cluster"]
                                if cl:
                                    self.clusters.setdefault(cl, {"title": cl, "items": []})
                                    if sid2 not in self.clusters[cl]["items"]:
                                        self.clusters[cl]["items"].append(sid2)

                                if key and key not in self.seen:
                                    self.seen.append(key)
                                updated.append(sid2)
                            else:
                                debug_emit({"ev":"slots_full"})
        for jid in rm:
            del self.jobs[jid]
        return updated

# ---------- Render
def render_slots_summary():
    slots = st.session_state.slots
    filled = len([s for s in slots.values() if (s.get("label") or "").strip()])
    st.progress(min(1.0, filled / 4), text=f"Progress: {filled}/4")

def render_history():
    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def parse_final_lines(text: str) -> List[str]:
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("- "):
            lines.append(ln[2:].strip())
        elif ln.lower().startswith("- label:"):
            lines.append(ln.split(":", 1)[-1].strip())
    return lines[:4]

def render_final_card(final_text: str, slots: Dict[str, Dict[str, Any]]):
    lines = parse_final_lines(final_text)
    for idx, sid in enumerate(["S1", "S2", "S3", "S4"]):
        if idx >= len(lines):
            break
        s = slots.get(sid)
        txt = lines[idx]
        img = (s.get("media", {}).get("best_image_url") or "") if s else ""
        col_text, col_img = st.columns([3, 2], vertical_alignment="center")
        with col_text:
            st.markdown(f"**{s.get('label','').split('—')[-1].strip() if s else 'Item'}**")
            st.write(txt)
        with col_img:
            if img:
                st.markdown(
                    f'''
                    <div style="display:flex;justify-content:center;">
                      <img src="{img}" alt="Expert Card item image"
                           style="width:100%;max-width:320px;height:auto;
                                  border-radius:12px;border:1px solid rgba(0,0,0,0.06);
                                  object-fit:contain;" />
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
            else:
                st.caption("(no image)")

def build_export_html(final_text: str, slots: Dict[str, Dict[str, Any]]) -> str:
    lines = parse_final_lines(final_text)
    items = []
    for idx, sid in enumerate(["S1","S2","S3","S4"]):
        if idx >= len(lines): break
        s = slots.get(sid, {})
        label = (s.get("label","") or "").split("—")[-1].strip() or f"Item {idx+1}"
        body  = lines[idx]
        img   = (s.get("media",{}).get("best_image_url") or "").strip()
        items.append({"title": label, "body": body, "img": img})

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Expert Card</title>
</head>
<body style="margin:0;padding:24px;background:#ffffff;color:#111111;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.5;">
  <div style="max-width:900px;margin:0 auto;">
    <h1 style="margin:0 0 16px 0;font-size:28px;">Expert Card</h1>
    <div style="font-size:14px;color:#555;margin:0 0 24px 0;">Generated from interview notes.</div>

    {"".join([
      f'''
      <section style="display:flex;gap:20px;align-items:flex-start;justify-content:space-between;margin:0 0 28px 0;flex-wrap:wrap;">
        <div style="flex: 1 1 56%;min-width:260px;">
          <h2 style="margin:0 0 8px 0;font-size:20px;">{item["title"]}</h2>
          <p style="margin:0;font-size:16px;">{item["body"]}</p>
        </div>
        <div style="flex: 1 1 38%;min-width:220px;display:flex;justify-content:center;">
          { (f'<img src="{item["img"]}" alt="{item["title"]}" style="max-width:100%;height:auto;border-radius:12px;border:1px solid rgba(0,0,0,0.06);object-fit:contain;" />') if item["img"] else '<div style="color:#999;font-size:13px;">(no image)</div>' }
        </div>
      </section>
      '''
      for item in items
    ])}
  </div>
</body>
</html>"""
    return html

# ---------- Flow helpers (bestehend)
def _count_items_from_flow() -> int:
    flow = st.session_state.flow
    done_keys: List[str] = flow.get("done_keys", [])
    current = flow.get("current_item_key")
    return len(done_keys) + (1 if current else 0)

def update_flow_with_detected(detected_key: Optional[str]):
    """Pivot-Logik: Wenn detected_key != current -> markiere current als done und setze current=detected_key.
       Ab 6 Items: Hard-Stop-Planung (Budget 2 Assistant-Antworten)."""
    if not detected_key:
        return
    flow = st.session_state.flow
    current = flow.get("current_item_key")
    done_keys: List[str] = flow.get("done_keys", [])

    if current is None:
        flow["current_item_key"] = detected_key
        st.session_state.flow = flow
        n = _count_items_from_flow()
        if n >= 6 and not st.session_state.get("hard_stop_pending"):
            st.session_state.hard_stop_pending = True
            st.session_state.hard_stop_budget = 2
            st.session_state.hard_stop_signaled = False
        return

    if detected_key == current:
        return

    # Pivot => current wird abgeschlossen
    if current not in done_keys:
        done_keys.append(current)
    flow["done_keys"] = done_keys
    flow["current_item_key"] = detected_key
    st.session_state.flow = flow

    # Reset Follow-up-Zähler für neues Item
    st.session_state.followup_count[detected_key] = 0

    n = _count_items_from_flow()
    if n >= 6 and not st.session_state.get("hard_stop_pending"):
        st.session_state.hard_stop_pending = True
        st.session_state.hard_stop_budget = 2
        st.session_state.hard_stop_signaled = False

# ---------- Main
st.set_page_config(page_title=APP_TITLE, page_icon="🟡", layout="wide")
st.title(APP_TITLE)
st.caption("Agent 2: Validator (artifact-aware) • Slots only with valid image")

# ---- Init state
init_state()

# ---- START-GATE (keine Modellaufrufe vor Klick)
st.info("Klicke **Start**, um das Interview zu beginnen. Bis dahin werden **keine** Modell-Aufrufe ausgeführt.")
start_col1, start_col2 = st.columns([1, 5])
with start_col1:
    if st.button("▶️ Start"):
        st.session_state.user_started = True
        st.rerun()

if not st.session_state.user_started:
    st.stop()  # Harte Bremse: beendet Run ohne Modelle / ohne Moduswahl

# --------- Mode selection (NACH Start, VOR Chat-UI) ----------
MODE_OPTIONS = ["Professional", "General / Lifescope"]
mode = st.radio(
    "Interview focus",
    MODE_OPTIONS,
    index=MODE_OPTIONS.index(st.session_state.get("agent1_mode", "Professional")),
    horizontal=True,
    help="Choose the style and breadth of Agent 1's conversation."
)
st.session_state["agent1_mode"] = mode

# Mode-Badge
badge_bg = "#E6F0FF" if "Professional" in mode else "#E9F9EE"
badge_fg = "#0A58CA" if "Professional" in mode else "#157347"
st.markdown(
    f"""
    <div style="margin:8px 0 4px 0;">
      <span style="
        display:inline-block;
        padding:4px 10px;
        border-radius:999px;
        background:{badge_bg};
        color:{badge_fg};
        font-weight:600;
        font-size:12px;
        border:1px solid rgba(0,0,0,0.08);
      ">Mode: {mode}</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ---- Orchestrator
orch = Orchestrator()

# ---- First opener from Agent 1 (nur nach Start, wenn History leer)
if not st.session_state.history:
    opener = agent1_stream_question([])  # Prompt ist modusabhängig; Opener-Hint aktiv
    st.session_state.history.append({"role": "assistant", "content": opener})
    st.rerun()

# ---- Agent 2 Poll (laufend)
orch.poll()

# ---- UI
render_slots_summary()
render_history()

# ---- FINAL CARD
if st.session_state.final_text:
    st.subheader("Your Expert Card")
    render_final_card(st.session_state.final_text, st.session_state.slots)

    export_html = build_export_html(st.session_state.final_text, st.session_state.slots)
    st.download_button(
        "⬇️ Export HTML",
        data=export_html.encode("utf-8"),
        file_name="expert-card.html",
        mime="text/html"
    )

# ---- Input handling
user_text = st.chat_input("Your turn…")
if user_text:
    # Log the user message
    st.session_state.history.append({"role": "user", "content": user_text})

    # (a) Korrekturen interpretieren und anwenden (Agent 1 Parser)
    corr = agent1_parse_corrections(user_text, st.session_state.slots)
    for act in corr.get("actions", []):
        action = (act.get("action") or act.get("type") or "").strip().lower()

        # Helper zum Finden eines Slots via slot_key oder (entity_type,entity_name)
        def _find_slot_by_action_key(a: Dict[str, Any]) -> Optional[str]:
            sk = (a.get("slot_key") or "").strip().lower()
            if sk:
                # sk kann 'person|gabor mate' sein oder eine Slot-ID
                if sk in st.session_state.slots:
                    return sk
                for sid0, s0 in st.session_state.slots.items():
                    if s0.get("key","") == sk:
                        return sid0
            et = (a.get("entity_type") or "").strip().lower()
            nm = (a.get("entity_name") or a.get("from_name") or "").strip()
            if et and nm:
                key0 = normalize_key(et, nm)
                for sid0, s0 in st.session_state.slots.items():
                    if s0.get("key","") == key0:
                        return sid0
            return None

        if action == "drop_item":
            sid_drop = _find_slot_by_action_key(act)
            if sid_drop and sid_drop in st.session_state.slots:
                cl = st.session_state.slots[sid_drop].get("cluster","")
                del st.session_state.slots[sid_drop]
                # Cluster-Index bereinigen
                if cl and cl in st.session_state.clusters:
                    st.session_state.clusters[cl]["items"] = [x for x in st.session_state.clusters[cl]["items"] if x != sid_drop]

        elif action == "reassign_cluster":
            sid_rc = _find_slot_by_action_key(act)
            new_cl = (act.get("cluster_id") or "").strip()
            if sid_rc and sid_rc in st.session_state.slots and new_cl:
                old = st.session_state.slots[sid_rc].get("cluster","")
                st.session_state.slots[sid_rc]["cluster"] = new_cl
                # Cluster-Index aktualisieren
                if old and old in st.session_state.clusters:
                    st.session_state.clusters[old]["items"] = [x for x in st.session_state.clusters[old]["items"] if x != sid_rc]
                st.session_state.clusters.setdefault(new_cl, {"title": new_cl, "items": []})
                if sid_rc not in st.session_state.clusters[new_cl]["items"]:
                    st.session_state.clusters[new_cl]["items"].append(sid_rc)

        elif action == "replace_item":
            # Beispiel: {"action":"replace_item","entity_type":"book","from_name":"hungry ghosts","to_name":"scattered minds"}
            et = (act.get("entity_type") or "").strip().lower()
            from_nm = (act.get("from_name") or "").strip()
            to_nm = (act.get("to_name") or "").strip()
            if et and to_nm:
                # Optional: bestehenden Slot finden, Cluster übernehmen
                sid_rep = _find_slot_by_action_key({"entity_type": et, "entity_name": from_nm})
                cluster_id = ""
                if sid_rep and sid_rep in st.session_state.slots:
                    cluster_id = st.session_state.slots[sid_rep].get("cluster","")
                    # alten Slot entfernen
                    old_cl = cluster_id
                    del st.session_state.slots[sid_rep]
                    if old_cl and old_cl in st.session_state.clusters:
                        st.session_state.clusters[old_cl]["items"] = [x for x in st.session_state.clusters[old_cl]["items"] if x != sid_rep]

                # Neue gezielte Suche planen (Artefakt nach Typ raten)
                artifact_guess = {
                    "person": "portrait",
                    "book": "book_cover",
                    "podcast": "podcast_cover",
                    "tool": "tool_logo",
                    "film": "film_poster",
                }.get(et, "portrait")
                # Query disambig (wenn Person vorhanden ist, anhängen)
                query = to_nm
                if et == "book":
                    # falls Cluster bekannt (z. B. 'gabor-mate'), setze "book cover"
                    if cluster_id:
                        query = f"{to_nm} book cover"
                    else:
                        query = f"{to_nm} book cover"
                st.session_state.pending_searches.append({
                    "entity_type": et,
                    "entity_name": to_nm,
                    "artifact": artifact_guess,
                    "search_query": query,
                    "cluster_id": cluster_id or _strip_accents(to_nm.lower()).replace(" ", "-")
                })

        elif action == "prefer_variant":
            # Beispiel: {"action":"prefer_variant","slot_key":"book|in the realm of hungry ghosts","rule":"no_text_on_cover"}
            sid_pref = _find_slot_by_action_key(act)
            rule = (act.get("rule") or "").strip().lower()
            if sid_pref and sid_pref in st.session_state.slots:
                s = st.session_state.slots[sid_pref]
                et = s.get("item_type","")
                nm = (s.get("label","").split("—")[-1] or "").strip()
                artifact = s.get("artifact","")
                if et and nm and artifact:
                    # Variantensuche (einfacher Query-Zusatz)
                    suffix = ""
                    if rule == "no_text_on_cover":
                        suffix = " no text minimal"
                    elif rule == "face_only":
                        suffix = " face only portrait"
                    query = f"{nm} {artifact.replace('_', ' ')}{suffix}".strip()
                    st.session_state.pending_searches.append({
                        "entity_type": et,
                        "entity_name": nm,
                        "artifact": artifact,
                        "search_query": query,
                        "cluster_id": s.get("cluster","") or _strip_accents(nm.lower()).replace(" ", "-")
                    })

    # (b) Gezielte Suchen planen (Agent 1 erzeugt artifact-aware Queries)
    last_q = ""
    for m in reversed(st.session_state.history[:-1]):
        if m["role"] == "assistant":
            last_q = m["content"]
            break

    planned = agent1_plan_searches(last_q, user_text, st.session_state.slots)
    if planned:
        st.session_state.pending_searches.extend(planned)

    # Agent 2 beobachten lassen (asynchron) – verarbeitet pending_searches + Validierung
    orch.schedule_watch(last_q, user_text)

    # *** Wichtig: Sofortiges Polling vor dem nächsten Agent-1-Call,
    # damit Enforcer/Follow-up-Zähler rechtzeitig injiziert werden. ***
    orch.poll()

    # Agent 1 — Streaming der nächsten Frage (Responses)
    nxt = agent1_stream_question(st.session_state.history)
    st.session_state.history.append({"role": "assistant", "content": nxt})

    # Finalize ONLY when explicit handoff phrase detected
    low = nxt.lower()
    if any(phrase in low for phrase in HANDOFF_PHRASES):
        st.session_state.final_text = agent3_finalize(st.session_state.history, st.session_state.slots)

    st.rerun()

# ---- Actions
c1, c2 = st.columns(2)
with c1:
    if st.button("✨ Finalize (manual)"):
        st.session_state.final_text = agent3_finalize(st.session_state.history, st.session_state.slots)
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
        # Sicherstellen, dass beim Neustart wieder das Start-Gate aktiv ist
        st.session_state["user_started"] = False
        st.rerun()
