# app.py — Expert Card Creator (GPT-5, Single-Agent mit versteckten ACTIONS)
# - Chat-Agent (Director) stellt Fragen, extrahiert Infos und emittiert versteckte ACTIONS
# - Background-Threads starten aus ACTIONS (Websuche-Simulation + Vision-Validierung via GPT-5)
# - Fortschritt, Debug-Panel, Finalizer
# - Keine Heuristiken: Extraktion 100% durch das LLM

import os
import re
import queue
import threading
from typing import List, Dict, Any, Tuple

import streamlit as st
from openai import OpenAI

# ─────────────────────────────────────────────
# Grundkonfiguration
# ─────────────────────────────────────────────
st.set_page_config(page_title="Expert Card Creator", page_icon="📝", layout="centered")
st.title("📝 Expert Card Creator")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("Set OPENAI_API_KEY in Streamlit → Settings → Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
CHAT_MODEL = os.getenv("DTBR_CHAT_MODEL", "gpt-5")  # ein Modell für alles

# Themen-Plan: Reihenfolge, Followups, erlaubte Actions
TOPICS_PLAN = [
    {"name": "Book",    "followups": 2, "actions": ["SEARCH_COVER"]},
    {"name": "Podcast", "followups": 2, "actions": ["SEARCH_PODCAST", "SEARCH_LOGO"]},
    {"name": "Person",  "followups": 1, "actions": ["SEARCH_PORTRAIT"]},
    {"name": "Tool",    "followups": 0, "actions": ["SEARCH_LOGO"]},
]

MAX_TOTAL_QUESTIONS = 10  # harter Deckel, damit es nicht endlos weitergeht

# Globale Queues/State für Background-Ergebnisse
RESULTS_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()

# ─────────────────────────────────────────────
# Session-Initialisierung
# ─────────────────────────────────────────────
def ensure_session():
    if "history" not in st.session_state:
        st.session_state.history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a warm, focused interviewer for a concise expert card. "
                    "Avoid redundant confirmations. Ask short, meaningful questions."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Hi! I'll help you craft a tiny expert card. "
                    "First, which book has helped you professionally? "
                    "Please give the title (author optional)."
                ),
            },
        ]
    if "profile" not in st.session_state:
        st.session_state.profile = {
            "topics": [
                {
                    "name": t["name"],
                    "status": "queued",
                    "followups": t["followups"],
                    "answers": [],
                    "media": {},
                }
                for t in TOPICS_PLAN
            ],
            "current_topic_index": 0,
            "bot_questions": 0,
            "finalizing": False,
        }
    if "inflight_keys" not in st.session_state:
        st.session_state.inflight_keys = set()
    if "results" not in st.session_state:
        st.session_state.results: List[Dict[str, Any]] = []
    if "final_output" not in st.session_state:
        st.session_state.final_output = None
    if "debug" not in st.session_state:
        st.session_state.debug: List[str] = []

ensure_session()

def log_debug(msg: str):
    st.session_state.debug.append(msg)

def current_topic() -> Dict[str, Any]:
    idx = st.session_state.profile["current_topic_index"]
    return st.session_state.profile["topics"][idx]

def advance_topic_if_needed():
    """Wenn Followups aufgebraucht → Status done → nächstes Topic (falls vorhanden)."""
    idx = st.session_state.profile["current_topic_index"]
    topic = st.session_state.profile["topics"][idx]
    if topic["followups"] <= 0 and topic["status"] != "done":
        topic["status"] = "done"
    if topic["status"] == "done" and idx + 1 < len(st.session_state.profile["topics"]):
        st.session_state.profile["current_topic_index"] += 1

def all_topics_completed() -> bool:
    return all(t["status"] == "done" for t in st.session_state.profile["topics"])

def reached_global_limits() -> bool:
    return (
        st.session_state.profile["bot_questions"] >= MAX_TOTAL_QUESTIONS
        or all_topics_completed()
    )

def finalize_if_possible():
    if not st.session_state.profile["finalizing"]:
        st.session_state.profile["finalizing"] = True
        log_debug("finalize_if_possible(): set finalizing=True")

# ─────────────────────────────────────────────
# Director-Agent (einziger Chat-Agent)
# ─────────────────────────────────────────────
DIRECTOR_INSTRUCTIONS = """
You are the Director Agent for a short expert-card interview.

SCOPE & ORDER (STRICT)
- Follow THIS exact topic order and never introduce others:
  1) Book — ask up to 2 short follow-ups
  2) Podcast — up to 2
  3) Person — up to 1
  4) Tool — 0 follow-ups (just collect a name if available)
- Move on when you have enough signal for the current topic (no confirmations unless truly ambiguous).

STYLE
- One concise, fresh question per turn.
- No repetitive phrasings; no “Is that correct?” unless ambiguous.
- Mirror user's language if clearly not English; otherwise use English.

HIDDEN ACTIONS (CRITICAL)
- When you have enough to launch background research for the current topic, emit a hidden block:
  [[ACTIONS]]
  ACTION: <TYPE> [title="..."] [author="..."] [name="..."]
  [[/ACTIONS]]
- Allowed actions:
  Book:    SEARCH_COVER (require title; author if available)
  Podcast: SEARCH_PODCAST or SEARCH_LOGO (require canonical podcast title)
  Person:  SEARCH_PORTRAIT (require canonical person name)
  Tool:    SEARCH_LOGO (require canonical brand/tool title)

EXTRACTION
- You must extract canonical titles/names directly from the user's latest answer.
- Do not pass full sentences as titles. Use proper-cased, concise names (e.g., “Escape to Rural France”, not the whole sentence).

STOPPING
- Stop asking when:
  (a) all topics are collected in the minimal scope, or
  (b) the global question budget is reached.
- When you believe enough content is collected across all topics, emit:
  [[ACTIONS]]
  ACTION: FINALIZE_READY
  [[/ACTIONS]]

OUTPUT FORMAT (MANDATORY)
1) First lines: the VISIBLE user-facing question (one or two short sentences).
2) Then, if any, a hidden ACTION block exactly in this form:
[[ACTIONS]]
ACTION: <TYPE> [title="..."] [author="..."] [name="..."]
[[/ACTIONS]]
Do NOT show actions in the visible question.
"""

ACTIONS_BLOCK_RE = re.compile(r"\[\[ACTIONS\]\](.*?)\[\[/ACTIONS\]\]", re.DOTALL | re.IGNORECASE)
ACTION_LINE_RE = re.compile(
    r'^\s*ACTION:\s*(?P<kind>SEARCH_COVER|SEARCH_PODCAST|SEARCH_LOGO|SEARCH_PORTRAIT|FINALIZE_READY)'
    r'(?:\s+title="(?P<title>[^"]*)")?'
    r'(?:\s+author="(?P<author>[^"]*)")?'
    r'(?:\s+name="(?P<name>[^"]*)")?\s*$',
    re.IGNORECASE
)

def split_visible_and_actions(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Zerlegt Assistant-Output in sichtbaren Teil + geparste ACTIONS; ACTION-Zeilen im sichtbaren entfernen."""
    actions: List[Dict[str, str]] = []
    t = text or ""

    # [[ACTIONS]] Block zuerst
    block_match = ACTIONS_BLOCK_RE.search(t)
    if block_match:
        block = block_match.group(1)
        visible = ACTIONS_BLOCK_RE.sub("", t).strip()
        for line in block.splitlines():
            m = ACTION_LINE_RE.match(line)
            if not m:
                continue
            gd = m.groupdict()
            act = {"kind": gd["kind"].upper()}
            if gd.get("title"):  act["title"]  = gd["title"].strip()
            if gd.get("author"): act["author"] = gd["author"].strip()
            if gd.get("name"):   act["name"]   = gd["name"].strip()
            actions.append(act)
        # evtl. übriggebliebene ACTION:-Zeilen aus visible entfernen
        visible = "\n".join([ln for ln in visible.splitlines()
                             if not ln.strip().upper().startswith("ACTION:")]).strip()
        return visible, actions

    # Kein Block → entferne einzelne ACTION-Zeilen sicherheitshalber
    visible_lines = []
    for ln in t.splitlines():
        if ACTION_LINE_RE.match(ln):
            gd = ACTION_LINE_RE.match(ln).groupdict()
            act = {"kind": gd["kind"].upper()}
            if gd.get("title"):  act["title"]  = gd["title"].strip()
            if gd.get("author"): act["author"] = gd["author"].strip()
            if gd.get("name"):   act["name"]   = gd["name"].strip()
            actions.append(act)
        else:
            visible_lines.append(ln)
    return "\n".join(visible_lines).strip(), actions

def director_turn() -> Tuple[str, List[Dict[str, str]]]:
    """Erzeuge nächste Frage + ACTIONS mit GPT-5."""
    topic = current_topic()
    sys = (
        f"{DIRECTOR_INSTRUCTIONS}\n\n"
        f"CURRENT TOPIC: {topic['name']}\n"
        f"REMAINING FOLLOW-UPS: {topic['followups']}\n"
        f"ALREADY SAID FOR THIS TOPIC: {topic['answers']}\n"
        f"TOTAL BOT QUESTIONS SO FAR: {st.session_state.profile['bot_questions']} (limit {MAX_TOTAL_QUESTIONS})\n"
        "Reminder: emit a hidden [[ACTIONS]] block whenever you have enough info to launch background search."
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=st.session_state.history + [{"role": "system", "content": sys}],
            temperature=0.6,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raw = f"(Director error: {e})"

    visible, acts = split_visible_and_actions(raw)
    if not visible:
        visible = f"One more on {topic['name'].lower()}: what makes it stand out for you?"
    return visible, acts

# ─────────────────────────────────────────────
# ACTION Guards & Background-Starter
# ─────────────────────────────────────────────
def allow_action(action: Dict[str, str]) -> bool:
    """Nur zum aktuellen Topic passende Actions zulassen, und minimal Felder prüfen."""
    kind = action.get("kind", "").upper()
    topic = current_topic()
    name = topic["name"]

    if kind == "FINALIZE_READY":
        return True  # Director entscheidet

    allowed = next((t["actions"] for t in TOPICS_PLAN if t["name"] == name), [])
    if kind not in set(allowed):
        return False

    # Minimalfelder strikt: wir vertrauen darauf, dass der Director kanonische Werte liefert
    if name == "Book" and kind == "SEARCH_COVER":
        return bool(action.get("title"))
    if name == "Podcast" and kind in ("SEARCH_PODCAST", "SEARCH_LOGO"):
        return bool(action.get("title"))
    if name == "Person" and kind == "SEARCH_PORTRAIT":
        return bool(action.get("name"))
    if name == "Tool" and kind == "SEARCH_LOGO":
        return bool(action.get("title"))

    return False

def start_background_action(action: Dict[str, str], topic_hint: str = ""):
    """Startet Nebenläufer. Key-Dedupe, Worker mit GPT-5 (Websuche-Simulation + Vision-Validierung)."""
    kind = action["kind"].upper()

    # Finalize-Flag
    if kind == "FINALIZE_READY":
        st.session_state.profile["finalizing"] = True
        log_debug("Action: FINALIZE_READY → finalizing=True")
        return

    # Topic-Zuordnung
    tk_name = topic_hint or current_topic()["name"]

    # kompakten Key bauen (max 80 chars)
    def norm_key(s: str) -> str:
        s = (s or "").strip().lower()
        return s[:80]

    # Mapping
    if kind in ("SEARCH_COVER", "SEARCH_LOGO", "SEARCH_PODCAST"):
        title = (action.get("title") or "").strip()
        author = (action.get("author") or "").strip()
        if not title and not author:
            log_debug(f"Reject action (missing title/author): {kind}")
            return
        if kind == "SEARCH_COVER":
            srch_kind = "book"
            key_id = norm_key(title or author)
        elif kind == "SEARCH_LOGO":
            srch_kind = "brand"
            key_id = norm_key(title)
        else:  # SEARCH_PODCAST
            srch_kind = "podcast"
            key_id = norm_key(title)
        key = f"{kind}:{key_id}"

        if key in st.session_state.inflight_keys:
            log_debug(f"Dedupe inflight: {key}")
            return
        st.session_state.inflight_keys.add(key)
        log_debug(f"Start background: {key} for topic={tk_name}")

        def worker():
            try:
                # 1) Websuche (LLM-simuliert)
                candidates = websearch_agent(srch_kind, title or author, author)

                # 2) Vision/Plausibilitätscheck
                validator = vision_validator_agent(
                    srch_kind,
                    expected_title=title or author,
                    expected_author_or_name=(author or title),
                    candidates=candidates,
                )
                best = validator.get("best_index", -1)
                chosen = candidates[best] if 0 <= best < len(candidates) else {}

                # Ergebnis ablegen
                RESULTS_QUEUE.put({
                    "key": key,
                    "topic_kind": tk_name,
                    "title": title,
                    "name": action.get("name", ""),
                    "image": chosen,
                    "candidates": candidates,
                    "validator": validator,
                })
            finally:
                # Key bleibt bis Harvest; wir räumen beim Merge auf
                pass

        threading.Thread(target=worker, daemon=True).start()
        return

    if kind == "SEARCH_PORTRAIT":
        name = (action.get("name") or "").strip()
        if not name:
            log_debug("Reject action (missing name) for SEARCH_PORTRAIT")
            return
        key = f"{kind}:{norm_key(name)}"
        if key in st.session_state.inflight_keys:
            log_debug(f"Dedupe inflight: {key}")
            return
        st.session_state.inflight_keys.add(key)
        log_debug(f"Start background: {key} for topic={tk_name}")

        def worker():
            try:
                candidates = websearch_agent("person", name, "")
                validator = vision_validator_agent(
                    "person",
                    expected_title=name,
                    expected_author_or_name=name,
                    candidates=candidates,
                )
                best = validator.get("best_index", -1)
                chosen = candidates[best] if 0 <= best < len(candidates) else {}
                RESULTS_QUEUE.put({
                    "key": key,
                    "topic_kind": tk_name,
                    "name": name,
                    "image": chosen,
                    "candidates": candidates,
                    "validator": validator,
                })
            finally:
                pass

        threading.Thread(target=worker, daemon=True).start()
        return

# ─────────────────────────────────────────────
# Websuche & Validator (LLM-basiert, Natural Language)
# ─────────────────────────────────────────────
def websearch_agent(kind: str, title_or_name: str, author: str = "") -> List[Dict[str, str]]:
    """
    LLM-simulierte Bildsuche (3 Kandidaten).
    Format (jede Zeile):
      CANDIDATE: image_url|page_url|source
    """
    sys = (
        "You are a web image search assistant. Return 1-3 plausible candidates.\n"
        "Format: each line 'CANDIDATE: image_url|page_url|source'. No extra text."
    )
    user = f"kind={kind}; query_title_or_name={title_or_name}; author={author}"
    try:
        r = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        text = (r.choices[0].message.content or "").strip()
    except Exception as e:
        text = f"(error {e})"

    candidates = []
    for line in text.splitlines():
        if not line.strip().lower().startswith("candidate:"):
            continue
        payload = line.split(":", 1)[1].strip()
        parts = [p.strip() for p in payload.split("|")]
        if len(parts) >= 1:
            image_url = parts[0]
            page_url = parts[1] if len(parts) >= 2 else ""
            source = parts[2] if len(parts) >= 3 else ""
            candidates.append({"image_url": image_url, "page_url": page_url, "source": source})
    return candidates[:3]

def vision_validator_agent(kind: str, expected_title: str, expected_author_or_name: str,
                           candidates: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Sehr einfacher Plausibilitäts-Validator (ohne Bilddownload):
    Antwortzeile:
      BEST: <0..2 or -1> | CONF: <0.0..1.0> | REASON: ...
    """
    header = (
        "You are a plausibility validator. Pick the best candidate index for the expected item.\n"
        "Answer in one line: BEST: <0..2 or -1> | CONF: <0.0..1.0> | REASON: <short>.\n"
        "If uncertain, pick -1 with low confidence."
    )
    desc = f"kind={kind}; expected_title={expected_title}; expected_author_or_name={expected_author_or_name}"
    listing = []
    for i, c in enumerate(candidates):
        listing.append(f"[{i}] image_url={c.get('image_url','')} page_url={c.get('page_url','')} source={c.get('source','')}")
    content = header + "\n" + desc + "\n" + "\n".join(listing)
    try:
        r = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": content}],
            temperature=0.1,
        )
        line = (r.choices[0].message.content or "").strip()
    except Exception as e:
        line = f"BEST: -1 | CONF: 0.0 | REASON: error {e}"

    best, conf, reason = -1, 0.0, ""
    m = re.search(r"BEST:\s*(-?\d+)", line)
    if m:
        try: best = int(m.group(1))
        except: best = -1
    m = re.search(r"CONF:\s*([0-9.]+)", line)
    if m:
        try: conf = float(m.group(1))
        except: conf = 0.0
    m = re.search(r"REASON:\s*(.*)$", line)
    if m:
        reason = m.group(1).strip()
    return {"best_index": best, "confidence": conf, "reason": reason, "raw": line}

# ─────────────────────────────────────────────
# Finalizer (nach Abschluss aller Topics + Recherchen)
# ─────────────────────────────────────────────
def finalizer_agent(profile: Dict[str, Any]) -> str:
    """
    Warmer, professioneller Abschluss (3–5 Sätze) + kompakte Liste je Topic mit One-Liner.
    Reine Sprache, kein JSON.
    """
    sys = (
        "You are a finalizer. Compose a concise, warm, professional summary (3–5 sentences) "
        "that presents the user positively without flattery. Then provide a compact list per item "
        "(Book/Podcast/Person/Tool) with a one-liner why it matters. Keep it crisp."
    )
    # Fakten als einfacher Text
    lines = []
    for t in profile["topics"]:
        if not t["answers"]:
            continue
        head = f"- {t['name']}:"
        facts = "; ".join([a.strip() for a in t["answers"] if a.strip()])
        chosen = t.get("media", {}).get("chosen", {})
        media_hint = chosen.get("image_url", "")
        if media_hint:
            lines.append(f"{head} {facts} (image: {media_hint})")
        else:
            lines.append(f"{head} {facts}")
    user = "Interview facts:\n" + "\n".join(lines) if lines else "Interview facts: (empty)"

    try:
        r = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            temperature=0.5,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(Finalizer error: {e})"

def all_research_done_for_answered_topics() -> bool:
    for t in st.session_state.profile["topics"]:
        if t["answers"]:
            if not t.get("media", {}).get("done"):
                return False
    return True

# ─────────────────────────────────────────────
# Chat-Eingabe
# ─────────────────────────────────────────────
user_text = st.chat_input("Your answer…")
if user_text:
    # 1) User-Message ins Log
    st.session_state.history.append({"role": "user", "content": user_text})

    # 2) Topic aktualisieren
    topic = current_topic()
    topic["answers"].append(user_text)
    if topic["followups"] > 0:
        topic["followups"] -= 1
    advance_topic_if_needed()

    # 3) Director antwortet (Frage + versteckte ACTIONS)
    if not st.session_state.profile["finalizing"] and not reached_global_limits():
        visible, actions = director_turn()

        # sichtbare Frage
        st.session_state.history.append({"role": "assistant", "content": visible})
        st.session_state.profile["bot_questions"] += 1

        # ACTIONS starten (nur gültige)
        for act in actions:
            if allow_action(act):
                start_background_action(act)

        # ggf. sofort Finalisierung anschalten (falls Director FINALIZE_READY geschickt hat)
        if reached_global_limits():
            finalize_if_possible()
            st.session_state.history.append({
                "role": "assistant",
                "content": "Thanks, I’ve got a solid picture now. I’ll assemble your expert card in the background."
            })
    else:
        # Bereits im Finalisieren → keine neuen Fragen mehr
        if not any(m["content"].endswith("in the background.") for m in st.session_state.history if m["role"] == "assistant"):
            st.session_state.history.append({
                "role": "assistant",
                "content": "Thanks, I’ve got a solid picture now. I’ll assemble your expert card in the background."
            })

    st.rerun()

# ─────────────────────────────────────────────
# Background-Ergebnisse ernten & Session mergen
# ─────────────────────────────────────────────
harvested = []
while True:
    try:
        harvested.append(RESULTS_QUEUE.get_nowait())
    except queue.Empty:
        break

def find_topic_index_by_name(name: str) -> int:
    for i, t in enumerate(st.session_state.profile["topics"]):
        if t["name"].lower() == (name or "").lower():
            return i
    return -1

for res in harvested:
    key = res.get("key", "")
    if key:
        st.session_state.inflight_keys.discard(key)

    kind = res.get("topic_kind", "")
    idx = find_topic_index_by_name(kind)
    if idx == -1:
        continue
    t = st.session_state.profile["topics"][idx]
    # Media mergen
    t["media"]["chosen"] = res.get("image", {})
    t["media"]["candidates"] = res.get("candidates", [])
    t["media"]["validator"] = res.get("validator", {})
    t["media"]["done"] = True

    # Ergebnisse deduplizieren & anzeigen
    img_url = (res.get("image") or {}).get("image_url", "")
    dup = any((r.get("image", {}).get("image_url") == img_url) and (r.get("topic_kind") == kind)
              for r in st.session_state.results)
    if not dup:
        st.session_state.results.append(res)
    log_debug(f"harvested: {kind} | image={img_url or '-'}")

# ─────────────────────────────────────────────
# Fortschritt / Status
# ─────────────────────────────────────────────
interview_done = sum(1 for t in st.session_state.profile["topics"] if t["status"] == "done")
research_done  = sum(1 for t in st.session_state.profile["topics"] if t.get("media", {}).get("done"))
progress_total = len(TOPICS_PLAN) * 2
progress_val = min(interview_done + research_done, progress_total)
st.progress(progress_val / progress_total, text=f"Progress: {progress_val}/{progress_total}")

if st.session_state.inflight_keys:
    st.info("🔎 Background research running… you can keep chatting.")

# ─────────────────────────────────────────────
# Chatverlauf (nur sichtbare Messages)
# ─────────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Debug
with st.expander("Debug"):
    st.write("In-flight:", list(st.session_state.inflight_keys) or "—")
    if st.session_state.results:
        st.write(f"Results count: {len(st.session_state.results)}")
    else:
        st.write("Results: —")
    if st.session_state.debug:
        for d in st.session_state.debug[-30:]:
            st.code(d, language="text")

# ─────────────────────────────────────────────
# Finalisierung (wenn: finalizing an ODER Limits erreicht) UND alle Recherchen fertig
# ─────────────────────────────────────────────
if (st.session_state.profile["finalizing"] or reached_global_limits()) \
        and not st.session_state.inflight_keys \
        and all_research_done_for_answered_topics():
    if not st.session_state.final_output:
        st.session_state.final_output = finalizer_agent(st.session_state.profile)

# Ergebnisse in kompakter Form
if st.session_state.results:
    st.subheader("Results")
    for res in st.session_state.results[::-1]:
        c1, c2 = st.columns([1, 1.4])
        with c1:
            img = res.get("image", {}).get("image_url")
            if img:
                st.image(img, use_container_width=True, caption=res.get("topic_kind", ""))
            else:
                st.write("(no image)")
        with c2:
            tk = res.get("topic_kind", "")
            title = res.get("title") or res.get("name") or ""
            st.markdown(f"**{tk}** — *{title}*")
            pg = res.get("image", {}).get("page_url", "")
            if pg:
                st.markdown(f"[Source]({pg})")

# Finaler Text
if st.session_state.final_output:
    st.subheader("Your Expert Card (Draft)")
    st.markdown(st.session_state.final_output)

# Controls
c1, c2 = st.columns(2)
with c1:
    if st.button("🔄 Restart"):
        for k in ["history", "profile", "inflight_keys", "results", "final_output", "debug"]:
            if k in st.session_state:
                del st.session_state[k]
        # Queue leeren
        while not RESULTS_QUEUE.empty():
            try:
                RESULTS_QUEUE.get_nowait()
            except queue.Empty:
                break
        ensure_session()
        st.rerun()
with c2:
    if st.button("🔁 Refresh"):
        st.rerun()
