# app.py — Expert Card Creator
# - Director mit festen Leitplanken & Limits
# - Unsichtbare [[ACTIONS]]-Blöcke (falls vorhanden)
# - Auto-Trigger für Hintergrundsuche (läuft auch ohne [[ACTIONS]])
# - Parallele Recherche + Vision-Validierung (LLM-basiert)
# - Harte Stopps & Finalizer

import os
import re
import queue
import threading
from typing import List, Dict, Any, Tuple

import streamlit as st
from openai import OpenAI

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Expert Card Creator", page_icon="📝", layout="centered")
st.title("📝 Expert Card Creator")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("Set OPENAI_API_KEY in Streamlit → Settings → Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Modelle
CHAT_MODEL = os.getenv("DTBR_CHAT_MODEL", "gpt-5")          # Director / Validator / Finalizer
SEARCH_MODEL = os.getenv("DTBR_SEARCH_MODEL", CHAT_MODEL)   # Websearch-Simulation (LLM)

# Gesprächs-Plan (Themen, Followups, Minimalfelder, erlaubte Actions)
TOPICS_PLAN = [
    {"name": "Book",    "followups": 2, "min_fields": ["title"], "actions": ["SEARCH_COVER"]},
    {"name": "Podcast", "followups": 2, "min_fields": ["title"], "actions": ["SEARCH_PODCAST", "SEARCH_LOGO"]},
    {"name": "Person",  "followups": 1, "min_fields": ["name"],  "actions": ["SEARCH_PORTRAIT"]},
    {"name": "Tool",    "followups": 0, "min_fields": ["title"], "actions": ["SEARCH_LOGO"]},
]

# Harte Grenzen fürs ganze Gespräch
MAX_TOTAL_QUESTIONS = 10
MAX_TOPICS = len(TOPICS_PLAN)

# Background queues/state (thread-safe)
RESULTS_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()

# ─────────────────────────────────────────────
# Session init
# ─────────────────────────────────────────────
def ensure_session():
    if "history" not in st.session_state:
        st.session_state.history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a warm, focused interviewer creating a concise 'expert card'. "
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
                {"name": t["name"], "status": "queued", "followups": t["followups"],
                 "answers": [], "media": {}, "research_started": False}
                for t in TOPICS_PLAN
            ],
            "current_topic_index": 0,
            "finalizing": False,
            "bot_questions": 0,
        }
    if "inflight_keys" not in st.session_state:
        st.session_state.inflight_keys = set()
    if "results" not in st.session_state:
        st.session_state.results: List[Dict[str, Any]] = []
    if "final_output" not in st.session_state:
        st.session_state.final_output = None
    if "progress_total" not in st.session_state:
        st.session_state.progress_total = len(TOPICS_PLAN) * 2  # Interview + Research je Topic
    if "progress_done" not in st.session_state:
        st.session_state.progress_done = 0
    if "debug" not in st.session_state:
        st.session_state.debug: List[str] = []

ensure_session()

# ─────────────────────────────────────────────
# Topic Helpers
# ─────────────────────────────────────────────
def log_debug(msg: str):
    st.session_state.debug.append(msg)

def current_topic() -> Dict[str, Any]:
    idx = st.session_state.profile["current_topic_index"]
    return st.session_state.profile["topics"][idx]

def find_topic_index_by_name(name: str) -> int:
    for i, t in enumerate(st.session_state.profile["topics"]):
        if t["name"].lower() == name.lower():
            return i
    return -1

def advance_topic_if_needed():
    """Wenn Followups aufgebraucht → Topic done → nächstes Thema wählen (falls vorhanden)."""
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
        if all_topics_completed():
            st.session_state.profile["finalizing"] = True
            log_debug("finalize_if_possible(): set finalizing=True (all topics completed)")

# ─────────────────────────────────────────────
# Director Prompt / ACTIONS Parsing
# ─────────────────────────────────────────────
DIRECTOR_INSTRUCTIONS = """
You are the Director Agent for a short expert-card interview.

SCOPE & ORDER
- You must follow this exact topic order and stop after the last one:
  1) Book → ask up to 2 follow-ups
  2) Podcast → up to 2
  3) Person → up to 1
  4) Tool → 0 follow-ups (just collect a name if available)
- Never introduce new topics outside this list.
- Per topic, ask only within that topic’s scope.

STYLE & BEHAVIOR
- One concise, fresh question per turn.
- Avoid repeating the same meaning.
- No “Is that correct?” confirmations unless truly ambiguous.
- Mirror the user's language if it's clearly not English; otherwise use English.
- After a topic reaches its follow-up limit or has enough substance, move on.

TRIGGERS (optional, hidden):
- If minimal info for the current topic is present, emit a hidden [[ACTIONS]] block with one action per line:
  [[ACTIONS]]
  ACTION: SEARCH_COVER title="..." author="..."
  [[/ACTIONS]]
- Allowed actions by topic:
  Book: SEARCH_COVER
  Podcast: SEARCH_PODCAST or SEARCH_LOGO
  Person: SEARCH_PORTRAIT
  Tool: SEARCH_LOGO
- Emit FINALIZE_READY only when all topics in scope have sufficient material.

OUTPUT FORMAT
1) First lines: the VISIBLE question (one or two short sentences).
2) Then, if any, a hidden block exactly like:
[[ACTIONS]]
ACTION: <TYPE> [title="..."] [author="..."] [name="..."]
[[/ACTIONS]]
Never mix actions into the visible question.
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
    """Extrahiert [[ACTIONS]]-Block, strippt ihn aus dem sichtbaren Text, parst Action-Zeilen.
       Fallback: ACTION:-Zeilen auch ohne Block entfernen.
    """
    actions: List[Dict[str, str]] = []

    # 1) [[ACTIONS]]...[[/ACTIONS]] block
    block_match = ACTIONS_BLOCK_RE.search(text or "")
    if block_match:
        block = block_match.group(1)
        visible = ACTIONS_BLOCK_RE.sub("", text).strip()
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
        # Sicherheit: im sichtbaren Teil evtl. hineingeratene ACTION-Zeilen entfernen
        visible = "\n".join([ln for ln in visible.splitlines() if not ln.strip().upper().startswith("ACTION:")]).strip()
        return visible, actions

    # 2) kein Block → entferne ACTION:-Zeilen aus sichtbarem Text
    visible_lines = []
    for ln in (text or "").splitlines():
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

def director_ask_next() -> Tuple[str, List[Dict[str, str]]]:
    """Frage vom Director generieren; sichtbarer Text + geparste Actions zurückgeben."""
    topic = current_topic()
    bot_q = st.session_state.profile.get("bot_questions", 0)

    sys = (
        f"{DIRECTOR_INSTRUCTIONS}\n\n"
        f"CURRENT TOPIC: {topic['name']}\n"
        f"REMAINING FOLLOW-UPS: {topic['followups']}\n"
        f"ALREADY SAID FOR THIS TOPIC: {topic['answers']}\n"
        f"TOTAL BOT QUESTIONS SO FAR: {bot_q} (hard limit {MAX_TOTAL_QUESTIONS})\n"
        "Remember: visible question first; optional [[ACTIONS]] block after."
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
# Guards: nur sinnvolle Actions zulassen
# ─────────────────────────────────────────────
def allow_action(action: Dict[str, str], last_user_text: str) -> bool:
    """Nur passende, ausreichend informierte Actions akzeptieren – strikt pro Topic."""
    kind = action.get("kind", "").upper()
    topic = current_topic()
    name = topic["name"]

    # Finalize kann immer – aber nur, wenn noch nicht im Finalisieren
    if kind == "FINALIZE_READY":
        return not st.session_state.profile["finalizing"]

    # Zulässige Actions laut Plan
    allowed = next((t["actions"] for t in TOPICS_PLAN if t["name"] == name), [])
    if kind not in set(allowed):
        return False

    # Minimalfelder je Topic/Action
    if name == "Book" and kind == "SEARCH_COVER":
        title = (action.get("title") or "").strip()
        if not title:
            return False
    elif name == "Podcast" and kind in ("SEARCH_PODCAST", "SEARCH_LOGO"):
        title = (action.get("title") or "").strip()
        if not title and not last_user_text.strip():
            return False
    elif name == "Person" and kind == "SEARCH_PORTRAIT":
        person = (action.get("name") or "").strip()
        if not person:
            return False
    elif name == "Tool" and kind == "SEARCH_LOGO":
        title = (action.get("title") or "").strip()
        if not title and not last_user_text.strip():
            return False

    # Nicht doppelt, falls bereits Media vorhanden
    if topic.get("media", {}).get("done"):
        return False

    return True

# ─────────────────────────────────────────────
# Simple Extractors & Auto-Trigger
# ─────────────────────────────────────────────
def extract_book_title_author(text: str) -> Tuple[str, str]:
    """Sehr einfache Heuristik: 'Title by Author' oder nur Title."""
    t = text.strip().strip('"“”')
    # Split on ' by ' (case-insensitive)
    m = re.split(r"\s+by\s+", t, flags=re.IGNORECASE)
    if len(m) >= 2:
        return m[0].strip(' "“”'), m[1].strip(' "“”')
    return t, ""

def maybe_trigger_auto_actions(last_user_text: str):
    """Falls Director keine [[ACTIONS]] sendet, aber wir genug Info haben → selber Background starten."""
    topic = current_topic()
    name = topic["name"]
    if topic.get("research_started"):
        return  # schon einmal gestartet

    # Buch
    if name == "Book":
        # wir schauen alle bisherigen Antworten zum Thema durch (erste Antwort reicht oft)
        content = " ".join(topic["answers"]).strip()
        if not content:
            return
        title, author = extract_book_title_author(content)
        if not title:
            return
        action = {"kind": "SEARCH_COVER", "title": title}
        if author:
            action["author"] = author
        # Starten & markieren
        start_background_action(action, topic_hint="Book")
        topic["research_started"] = True
        log_debug(f"auto-trigger Book: {title} | {author}")
        return

    # Podcast
    if name == "Podcast":
        if topic["answers"]:
            title = topic["answers"][0].strip()
            if title:
                action = {"kind": "SEARCH_PODCAST", "title": title}
                start_background_action(action, topic_hint="Podcast")
                topic["research_started"] = True
                log_debug(f"auto-trigger Podcast: {title}")
        return

    # Person
    if name == "Person":
        if topic["answers"]:
            person = topic["answers"][0].strip()
            if person:
                action = {"kind": "SEARCH_PORTRAIT", "name": person}
                start_background_action(action, topic_hint="Person")
                topic["research_started"] = True
                log_debug(f"auto-trigger Person: {person}")
        return

    # Tool
    if name == "Tool":
        if topic["answers"]:
            title = topic["answers"][0].strip()
            if title:
                action = {"kind": "SEARCH_LOGO", "title": title}
                start_background_action(action, topic_hint="Tool")
                topic["research_started"] = True
                log_debug(f"auto-trigger Tool: {title}")
        return

# ─────────────────────────────────────────────
# Background agents (Websearch + Validation) – Natural Language
# ─────────────────────────────────────────────
def websearch_agent(kind: str, title_or_name: str, author: str = "") -> List[Dict[str, str]]:
    """
    LLM-simulierte Bildsuche. Format:
    CANDIDATE: image_url|page_url|source
    """
    sys = (
        "You are a web image search assistant. Return 1-3 plausible candidates.\n"
        "Format: each line 'CANDIDATE: image_url|page_url|source'. No extra text."
    )
    user = f"kind={kind}; query_title_or_name={title_or_name}; author={author}"
    try:
        r = client.chat.completions.create(
            model=SEARCH_MODEL,
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
    Plausibilitätswahl. Antwort in einer Zeile:
    BEST: <0..2 or -1> | CONF: <0.0..1.0> | REASON: ...
    """
    header = (
        "You are a vision/plausibility validator. Pick the best candidate index for the expected item.\n"
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

def start_background_action(action: Dict[str, str], topic_hint: str = ""):
    """Startet nebenläufige Suche/Validierung oder setzt Finalize-Flag. Dedup über inflight_keys.
       topic_hint erlaubt Zuordnung, falls current_topic sich inzwischen geändert hat.
    """
    kind = action["kind"].upper()

    if kind == "FINALIZE_READY":
        st.session_state.profile["finalizing"] = True
        log_debug("Action: FINALIZE_READY → set finalizing=True")
        return

    # Zieltopic bestimmen
    if topic_hint:
        tk_name = topic_hint
    else:
        tk_name = current_topic()["name"]

    # Map Action → key + worker
    if kind in ("SEARCH_COVER", "SEARCH_LOGO", "SEARCH_PODCAST"):
        title = (action.get("title") or "").strip()
        author = (action.get("author") or "").strip()
        if kind == "SEARCH_COVER":
            srch_kind = "book"
        elif kind == "SEARCH_LOGO":
            srch_kind = "brand"
        else:  # SEARCH_PODCAST
            srch_kind = "podcast"

        key = f"{kind}:{(title or author).lower()}"
        if not (title or author):
            log_debug(f"Action rejected (missing title/author): {kind}")
            return
        if key in st.session_state.inflight_keys:
            log_debug(f"Action dedup (already inflight): {key}")
            return
        st.session_state.inflight_keys.add(key)
        log_debug(f"Action start: {key} for topic={tk_name}")

        def worker():
            try:
                cands = websearch_agent(srch_kind, title or author, author)
                val = vision_validator_agent(srch_kind,
                                             expected_title=title or author,
                                             expected_author_or_name=(author or title),
                                             candidates=cands)
                best = val.get("best_index", -1)
                chosen = cands[best] if 0 <= best < len(cands) else {}
                RESULTS_QUEUE.put({
                    "key": key,
                    "topic_kind": tk_name,
                    "title": title,
                    "image": chosen,
                    "candidates": cands,
                    "validator": val
                })
            finally:
                pass

        threading.Thread(target=worker, daemon=True).start()
        return

    if kind == "SEARCH_PORTRAIT":
        name = (action.get("name") or "").strip()
        key = f"{kind}:{name.lower()}"
        if not name:
            log_debug("Action rejected (missing name) for SEARCH_PORTRAIT")
            return
        if key in st.session_state.inflight_keys:
            log_debug(f"Action dedup (already inflight): {key}")
            return
        st.session_state.inflight_keys.add(key)
        log_debug(f"Action start: {key} for topic={tk_name}")

        def worker():
            try:
                cands = websearch_agent("person", name)
                val = vision_validator_agent("person",
                                             expected_title=name,
                                             expected_author_or_name=name,
                                             candidates=cands)
                best = val.get("best_index", -1)
                chosen = cands[best] if 0 <= best < len(cands) else {}
                RESULTS_QUEUE.put({
                    "key": key,
                    "topic_kind": tk_name,
                    "name": name,
                    "image": chosen,
                    "candidates": cands,
                    "validator": val
                })
            finally:
                pass

        threading.Thread(target=worker, daemon=True).start()
        return

# ─────────────────────────────────────────────
# Handle user turn
# ─────────────────────────────────────────────
user_text = st.chat_input("Your answer…")
if user_text:
    # 1) Save user message
    st.session_state.history.append({"role": "user", "content": user_text})

    # 2) Update topic (store answer & budget)
    topic = current_topic()
    topic["answers"].append(user_text)
    if topic["followups"] > 0:
        topic["followups"] -= 1
    # ggf. Topic abschließen & weiter
    advance_topic_if_needed()

    # 2.5) Auto-Trigger: Wenn ausreichend Info vorhanden, sofort Background starten
    try:
        maybe_trigger_auto_actions(user_text)
    except Exception as e:
        log_debug(f"auto-trigger error: {e}")

    # 3) Ask Director (visible + actions) – nur wenn nicht bereits finalisieren und Limits nicht erreicht
    if not st.session_state.profile["finalizing"] and not reached_global_limits():
        visible, acts = director_ask_next()

        # Sichtbare Frage in den Chat
        st.session_state.history.append({"role": "assistant", "content": visible})
        # Bot-Frage zählen
        st.session_state.profile["bot_questions"] += 1

        # 4) Actions nur starten, wenn Guards OK
        for act in acts:
            if allow_action(act, user_text):
                start_background_action(act)

        # Falls jetzt globale Grenzen erreicht → finalisieren
        if reached_global_limits():
            finalize_if_possible()
            st.session_state.history.append({
                "role": "assistant",
                "content": "Thanks, I’ve got a solid picture now. I’ll assemble your expert card in the background."
            })
    else:
        # Bereits im Finalisieren → keine neue Frage mehr; ggf. einmalig Hinweis
        if not any(m["content"].endswith("in the background.") for m in st.session_state.history if m["role"] == "assistant"):
            st.session_state.history.append({
                "role": "assistant",
                "content": "Thanks, I’ve got a solid picture now. I’ll assemble your expert card in the background."
            })

    st.rerun()

# ─────────────────────────────────────────────
# Harvest background results & integrate into session
# ─────────────────────────────────────────────
harvested = []
while True:
    try:
        harvested.append(RESULTS_QUEUE.get_nowait())
    except queue.Empty:
        break

for res in harvested:
    key = res.get("key", "")
    if key:
        st.session_state.inflight_keys.discard(key)
    kind = res.get("topic_kind")
    # Ordne Ergebnis dem passenden Topic-Slot zu
    idx = find_topic_index_by_name(kind)
    if idx == -1:
        continue
    t = st.session_state.profile["topics"][idx]
    t["media"]["chosen"] = res.get("image", {})
    t["media"]["candidates"] = res.get("candidates", [])
    t["media"]["validator"] = res.get("validator", {})
    t["media"]["done"] = True

    # Ergebnisliste (Anzeige) deduplizieren
    img_url = (res.get("image") or {}).get("image_url", "")
    dup = any((r.get("image", {}).get("image_url") == img_url) and (r.get("topic_kind") == kind)
              for r in st.session_state.results)
    if not dup:
        st.session_state.results.append(res)
    log_debug(f"harvested: {kind} | image={img_url or '-'}")

# ─────────────────────────────────────────────
# Progress bar (Interview + Research)
# ─────────────────────────────────────────────
interview_done = sum(1 for t in st.session_state.profile["topics"] if t["status"] == "done")
research_done  = sum(1 for t in st.session_state.profile["topics"] if t.get("media", {}).get("done"))
progress = min(interview_done + research_done, st.session_state.progress_total)
st.progress(progress / st.session_state.progress_total, text=f"Progress: {progress}/{st.session_state.progress_total}")

if st.session_state.inflight_keys:
    st.info("🔎 Background research running… you can keep chatting.")

# ─────────────────────────────────────────────
# Render chat history (visible only)
# ─────────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Debugpaneel (optional einklappbar)
with st.expander("Debug"):
    if st.session_state.inflight_keys:
        st.write("In-flight:", list(st.session_state.inflight_keys))
    else:
        st.write("In-flight: —")
    if st.session_state.results:
        st.write(f"Results count: {len(st.session_state.results)}")
    else:
        st.write("Results: —")
    if st.session_state.debug:
        for d in st.session_state.debug[-30:]:
            st.code(d, language="text")

# ─────────────────────────────────────────────
# Finalizer trigger & rendering
# ─────────────────────────────────────────────
def all_research_done_for_answered_topics() -> bool:
    for t in st.session_state.profile["topics"]:
        if t["answers"]:
            if not t.get("media", {}).get("done"):
                return False
    return True

def finalizer_agent(profile: Dict[str, Any]) -> str:
    """
    Freundlicher, professioneller Abschluss + kompakte Liste je Topic.
    Reine Sprache, kein JSON.
    """
    sys = (
        "You are a finalizer. Compose a concise, warm, professional summary (3–5 sentences) "
        "that presents the user positively without flattery. Then provide a compact list per item "
        "(Book/Podcast/Person/Tool) with a one-liner why it matters. Keep it crisp."
    )
    # Profil als einfacher Text
    lines = []
    for t in profile["topics"]:
        if not t["answers"]:
            continue
        head = f"- {t['name']}:"
        facts = "; ".join([a.strip() for a in t["answers"] if a.strip()])
        chosen = t.get("media", {}).get("chosen", {})
        media_hint = chosen.get("image_url", "")
        lines.append(f"{head} {facts} (image: {media_hint})")
    user = "Interview facts:\n" + "\n".join(lines)

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

# Finalisieren nur, wenn Director „bereit“ signalisiert hat ODER Limits erreicht, und alle Recherchen fertig sind
if (st.session_state.profile["finalizing"] or reached_global_limits()) \
        and not st.session_state.inflight_keys \
        and all_research_done_for_answered_topics():
    if not st.session_state.final_output:
        st.session_state.final_output = finalizer_agent(st.session_state.profile)

# Ergebnisse (Bilder + Kurzinfos)
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
        for k in ["history", "profile", "inflight_keys", "results", "final_output",
                  "progress_total", "progress_done", "debug"]:
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
