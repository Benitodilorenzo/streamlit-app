# app.py — Expert Card Creator (Natural-Language Multi-Agents, invisible ACTIONS, guarded)
# Director decides; hidden [[ACTIONS]] block; parallel research; guarded triggers; finalizer after ready.

import os
import re
import time
import queue
import threading
from typing import List, Dict, Any, Optional

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

CHAT_MODEL = os.getenv("DTBR_CHAT_MODEL", "gpt-5")      # director / validator / finalizer
SEARCH_MODEL = os.getenv("DTBR_SEARCH_MODEL", CHAT_MODEL)  # websearch (hier via LLM ohne Tools)

TOPICS_ORDER = ["Book", "Podcast", "Person", "Tool"]
FOLLOWUP_BUDGET = {"Book": 2, "Podcast": 2, "Person": 1, "Tool": 0}

# Für jedes Topic: welche Action-Typen sind überhaupt sinnvoll/erlaubt?
ALLOWED_ACTIONS_BY_TOPIC = {
    "Book": {"SEARCH_COVER"},
    "Podcast": {"SEARCH_PODCAST", "SEARCH_LOGO"},
    "Person": {"SEARCH_PORTRAIT"},
    "Tool": {"SEARCH_LOGO"},
}

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
                {"name": name, "status": "queued", "followups": FOLLOWUP_BUDGET[name],
                 "answers": [], "media": {}, "research_started": False}
                for name in TOPICS_ORDER
            ],
            "current_topic_index": 0,
            "finalizing": False  # wird vom Director via ACTION signalisiert
        }
    if "inflight_keys" not in st.session_state:
        st.session_state.inflight_keys = set()         # aktive Jobs (Dedup)
    if "results" not in st.session_state:
        st.session_state.results: List[Dict[str, Any]] = []   # gesammelte Resultate
    if "final_output" not in st.session_state:
        st.session_state.final_output = None
    if "progress_total" not in st.session_state:
        st.session_state.progress_total = len(TOPICS_ORDER) * 2  # Interview + Research je Topic
    if "progress_done" not in st.session_state:
        st.session_state.progress_done = 0

ensure_session()

# ─────────────────────────────────────────────
# Topic Helpers
# ─────────────────────────────────────────────
def current_topic() -> Dict[str, Any]:
    idx = st.session_state.profile["current_topic_index"]
    return st.session_state.profile["topics"][idx]

def advance_topic_if_needed():
    """Wenn Followups aufgebraucht → Topic done → nächstes Thema wählen (falls vorhanden)."""
    idx = st.session_state.profile["current_topic_index"]
    topic = st.session_state.profile["topics"][idx]
    if topic["followups"] <= 0 and topic["status"] != "done":
        topic["status"] = "done"
    if topic["status"] == "done" and idx + 1 < len(st.session_state.profile["topics"]):
        st.session_state.profile["current_topic_index"] += 1

# ─────────────────────────────────────────────
# Director Prompt / ACTIONS Parsing
# ─────────────────────────────────────────────
DIRECTOR_INSTRUCTIONS = """
You are the Director Agent for an expert card interview.

RULES
- Default to English; mirror the user's language if it is clearly not English.
- Ask ONE concise, fresh question that progresses the current topic only.
- Avoid repeating questions with the same meaning.
- Keep a gentle, professional tone. Do not ask for confirmations like “Is that correct?” unless truly ambiguous.
- Roughly 1–2 follow-ups per topic; then transition naturally to the next topic.
- Only propose background actions for items the user actually mentioned (no guessing).
- Allowed action types per topic:
  Book → SEARCH_COVER
  Podcast → SEARCH_PODCAST or SEARCH_LOGO
  Person → SEARCH_PORTRAIT
  Tool → SEARCH_LOGO
- Trigger actions only when minimal info is present:
  • SEARCH_COVER: at least a title; author optional but helpful.
  • SEARCH_PODCAST / SEARCH_LOGO: podcast/channel/tool name present.
  • SEARCH_PORTRAIT: person name present.
- If you believe the entire interview has enough material to finalize, also emit: FINALIZE_READY.

OUTPUT FORMAT
1) First lines: the VISIBLE chat question (one or two short sentences).
2) Then, if any, a hidden block:
[[ACTIONS]]
ACTION: <TYPE> [title="..."] [author="..."] [name="..."]
(one action per line)
[[/ACTIONS]]

No markdown fences beyond the [[ACTIONS]] block. No JSON. The VISIBLE text must not include actions.
"""

ACTIONS_BLOCK_RE = re.compile(r"\[\[ACTIONS\]\](.*?)\[\[/ACTIONS\]\]", re.DOTALL | re.IGNORECASE)
ACTION_LINE_RE = re.compile(
    r'^\s*ACTION:\s*(?P<kind>SEARCH_COVER|SEARCH_PODCAST|SEARCH_LOGO|SEARCH_PORTRAIT|FINALIZE_READY)'
    r'(?:\s+title="(?P<title>[^"]*)")?'
    r'(?:\s+author="(?P<author>[^"]*)")?'
    r'(?:\s+name="(?P<name>[^"]*)")?\s*$',
    re.IGNORECASE
)

def split_visible_and_actions(text: str) -> (str, List[Dict[str, str]]):
    """Extrahiert [[ACTIONS]]-Block, strippt ihn aus dem sichtbaren Text, parst Action-Zeilen.
       Fällt zurück auf Einzelzeilen mit 'ACTION:' falls kein Block benutzt wurde.
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
        # zusätzlich: evtl. versehentlich im sichtbaren Teil verbliebene ACTION-Zeilen entfernen
        visible = "\n".join([ln for ln in visible.splitlines() if not ln.strip().upper().startswith("ACTION:")]).strip()
        return visible, actions

    # 2) kein Block → entferne Zeilen, die mit ACTION: beginnen
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

def director_ask_next() -> (str, List[Dict[str, str]]):
    """Frage vom Director generieren; sichtbarer Text + geparste Actions zurückgeben."""
    topic = current_topic()
    sys = (
        f"{DIRECTOR_INSTRUCTIONS}\n\n"
        f"CURRENT TOPIC: {topic['name']}\n"
        f"REMAINING FOLLOW-UPS: {topic['followups']}\n"
        f"ALREADY SAID FOR THIS TOPIC: {topic['answers']}\n"
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
    # Fallback, falls Modell nichts Sinnvolles liefert
    if not visible:
        visible = f"One more on {topic['name'].lower()}: what makes it stand out for you?"
    return visible, acts

# ─────────────────────────────────────────────
# Guards: Nur sinnvolle Actions zulassen
# ─────────────────────────────────────────────
def allow_action(action: Dict[str, str], last_user_text: str) -> bool:
    """Harter Programm-Guard: nur passende, ausreichend informierte Actions akzeptieren."""
    kind = action.get("kind", "").upper()
    topic = current_topic()
    name = topic["name"]

    # 1) Nur Actions erlauben, die zum aktuellen Topic passen
    if kind not in ALLOWED_ACTIONS_BY_TOPIC.get(name, set()):
        return False

    # 2) Minimalinformationen prüfen je nach Action-Typ
    if kind == "SEARCH_COVER":
        # Braucht mind. einen Titel
        title = (action.get("title") or "").strip()
        if not title:
            return False
        # Wenn das Topic bereits recherchiert wurde, nicht doppelt
        if topic.get("media", {}).get("done"):
            return False
        return True

    if kind in ("SEARCH_PODCAST", "SEARCH_LOGO"):
        title = (action.get("title") or "").strip()
        # Für Podcast/Logo braucht es zumindest einen Namen/Hinweis
        if not title and not last_user_text.strip():
            return False
        if topic.get("media", {}).get("done"):
            return False
        return True

    if kind == "SEARCH_PORTRAIT":
        person = (action.get("name") or "").strip()
        if not person:
            return False
        if topic.get("media", {}).get("done"):
            return False
        return True

    if kind == "FINALIZE_READY":
        # Director bestimmt das, kein weiterer Guard nötig
        return True

    return False

# ─────────────────────────────────────────────
# Background agents (websearch + validation) — natural language outputs
# ─────────────────────────────────────────────
def websearch_agent(kind: str, title_or_name: str, author: str = "") -> List[Dict[str, str]]:
    """
    LLM simulierte Bildsuche. Format:
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
    Vision/Plausibilitätswahl. Format:
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

def start_background_action(action: Dict[str, str]):
    """Startet nebenläufige Suche/Validierung oder setzt Finalize-Flag. Dedup über inflight_keys."""
    kind = action["kind"].upper()

    if kind == "FINALIZE_READY":
        st.session_state.profile["finalizing"] = True
        return

    # Map Action → key + worker
    if kind in ("SEARCH_COVER", "SEARCH_LOGO", "SEARCH_PODCAST"):
        title = (action.get("title") or "").strip()
        author = (action.get("author") or "").strip()
        if kind == "SEARCH_COVER":
            topic_kind = "Book"; srch_kind = "book"
        elif kind == "SEARCH_LOGO":
            # kann für Tool/Podcast genutzt werden; hier behandeln wir es generisch als 'brand'
            topic_kind = current_topic()["name"]  # bleibt beim aktuellen Topic
            srch_kind = "brand"
        else:  # SEARCH_PODCAST
            topic_kind = "Podcast"; srch_kind = "podcast"

        key = f"{kind}:{title.lower() or author.lower()}"
        if key in st.session_state.inflight_keys:
            return
        st.session_state.inflight_keys.add(key)

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
                    "topic_kind": topic_kind,
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
        if not name or key in st.session_state.inflight_keys:
            return
        st.session_state.inflight_keys.add(key)

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
                    "topic_kind": "Person",
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

    # 3) Ask Director (visible + actions)
    visible, acts = director_ask_next()
    # Sichtbare Frage in den Chat, ohne ACTIONS
    st.session_state.history.append({"role": "assistant", "content": visible})

    # 4) Actions nur starten, wenn Guards OK
    for act in acts:
        if allow_action(act, user_text):
            start_background_action(act)

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
    idx = None
    for i, t in enumerate(st.session_state.profile["topics"]):
        if t["name"] == kind:
            idx = i; break
    if idx is None:
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

# ─────────────────────────────────────────────
# Progress bar (Interview + Research)
# ─────────────────────────────────────────────
interview_done = sum(1 for t in st.session_state.profile["topics"] if t["status"] == "done")
research_done = sum(1 for t in st.session_state.profile["topics"] if t.get("media", {}).get("done"))
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

# Finalisieren nur, wenn Director „bereit“ signalisiert hat und alle Recherchen fertig sind
if st.session_state.profile["finalizing"] and not st.session_state.inflight_keys and all_research_done_for_answered_topics():
    if not st.session_state.final_output:
        st.session_state.final_output = finalizer_agent(st.session_state.profile)

# Ergebnisse (Bilder + Kurzinfos) zeigen
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
                  "progress_total", "progress_done"]:
            if k in st.session_state:
                del st.session_state[k]
        # Queue leeren
        while not RESULTS_QUEUE.empty():
            try: RESULTS_QUEUE.get_nowait()
            except queue.Empty: break
        ensure_session()
        st.rerun()
with c2:
    if st.button("🔁 Refresh"):
        st.rerun()
