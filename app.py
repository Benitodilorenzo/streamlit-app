# app.py — Expert Card Creator (Natural-Language Multi-Agents, no JSON)
# Director decides; background research & validation in parallel; finalizer runs when ready.

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

CHAT_MODEL = os.getenv("DTBR_CHAT_MODEL", "gpt-5")  # director / validator / finalizer
SEARCH_MODEL = os.getenv("DTBR_SEARCH_MODEL", CHAT_MODEL)  # websearch (wir nutzen ebenfalls Chat, ohne Tools)

TOPICS_ORDER = ["Book", "Podcast", "Person", "Tool"]
FOLLOWUP_BUDGET = {"Book": 2, "Podcast": 2, "Person": 1, "Tool": 0}

# Background queues/state (thread-safe, kein direkter Session-Zugriff in Threads)
RESULTS_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()
INFLIGHT_KEYS = set()   # auf Session gespiegelt, wir führen zusätzlich lokal für Sichtbarkeit
TASKS_LOCK = threading.Lock()


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
            "finalizing": False  # wird vom Director signalisiert
        }
    if "inflight_keys" not in st.session_state:
        st.session_state.inflight_keys = set()
    if "results" not in st.session_state:
        st.session_state.results: List[Dict[str, Any]] = []  # fertige Resultate (für Anzeige)
    if "final_output" not in st.session_state:
        st.session_state.final_output = None
    if "progress_steps" not in st.session_state:
        # total: pro Topic (Interview + Research) = 2
        st.session_state.progress_total = len(TOPICS_ORDER) * 2
        st.session_state.progress_done = 0  # wird dynamisch berechnet


ensure_session()


# ─────────────────────────────────────────────
# Utility: topic helpers
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
# Director prompt & parsing (natural language + action lines)
# ─────────────────────────────────────────────
DIRECTOR_INSTRUCTIONS = """
You are the Director Agent for an expert card interview.

RULES:
- Default to English; mirror the user's language if it is clearly not English.
- Current topic is provided in the extra system instructions appended below.
- Ask ONE concise, fresh question that progresses the interview.
- Avoid repeating the same question with the same meaning.
- ~1–2 follow-ups per topic; then move on naturally to the next topic.
- Do NOT ask for confirmations like “Is that correct?” unless there is genuine ambiguity.
- If the current topic already has enough information to trigger background work,
  output an ACTION line at the very end (after your question), one per line, for example:
  ACTION: SEARCH_COVER title="Antifragility" author="Nassim Taleb"
  ACTION: SEARCH_PORTRAIT name="Gabor Maté"
  ACTION: SEARCH_LOGO title="Secular Talk"
- If you believe the overall interview is sufficient to finalize, also add:
  ACTION: FINALIZE_READY

FORMAT:
- Your visible chat question first (one or two short sentences).
- Then (optionally) one or more ACTION lines, each on its own line, starting with 'ACTION:'.
- No JSON. No markdown fences.
"""

ACTION_RE = re.compile(
    r'^ACTION:\s*(?P<kind>SEARCH_COVER|SEARCH_PORTRAIT|SEARCH_LOGO|FINALIZE_READY)'
    r'(?:\s+title="(?P<title>[^"]*)")?'
    r'(?:\s+author="(?P<author>[^"]*)")?'
    r'(?:\s+name="(?P<name>[^"]*)")?',
    re.IGNORECASE
)

def parse_actions(text: str) -> List[Dict[str, str]]:
    actions = []
    for line in text.splitlines():
        m = ACTION_RE.match(line.strip())
        if not m:
            continue
        gd = m.groupdict()
        kind = m.group("kind").upper()
        act = {"kind": kind}
        if gd.get("title"):
            act["title"] = gd["title"].strip()
        if gd.get("author"):
            act["author"] = gd["author"].strip()
        if gd.get("name"):
            act["name"] = gd["name"].strip()
        actions.append(act)
    return actions

def director_ask_next() -> str:
    """Frage vom Director generieren; addiere Systemhinweis zum aktuellen Topic."""
    topic = current_topic()
    sys = (
        f"{DIRECTOR_INSTRUCTIONS}\n\n"
        f"CURRENT TOPIC: {topic['name']}\n"
        f"REMAINING FOLLOW-UPS FOR CURRENT TOPIC: {topic['followups']}\n"
        f"PREVIOUS ANSWERS FOR THIS TOPIC (if any): {topic['answers']}\n"
        "Remember: visible question first; optional ACTION lines at the end."
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=st.session_state.history + [{"role": "system", "content": sys}],
            temperature=0.6,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(Director error: {e})"


# ─────────────────────────────────────────────
# Background agents (websearch + validation) — natural language outputs
# ─────────────────────────────────────────────
def websearch_agent(kind: str, title_or_name: str, author: str = "") -> List[Dict[str, str]]:
    """
    Fragt das Modell nach 1–3 plausiblen Kandidaten. Reine Sprache, einfach parsbar:
    Bitte im Output Zeilen wie:
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
    Bitten wir das Modell, mit Vision (falls verfügbar) bzw. Plausibilität zu wählen:
    Antwortformat: "BEST: <index 0..2> | CONF: 0.0..1.0 | REASON: ..."
    Dazu geben wir vorher alle Kandidaten aufgelistet (als Text + Bild-URLs).
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
    """
    Startet eine Suche/Validierung/Finalisierung nebenläufig.
    Dedup über topic_key.
    """
    kind = action["kind"].upper()

    if kind == "FINALIZE_READY":
        st.session_state.profile["finalizing"] = True
        return

    if kind in ("SEARCH_COVER", "SEARCH_LOGO"):
        # item based on title (+ optional author)
        title = action.get("title", "").strip()
        author = action.get("author", "").strip()
        topic_kind = "Book" if kind == "SEARCH_COVER" else "Tool" if kind == "SEARCH_LOGO" else "Podcast"
        if kind == "SEARCH_COVER" and not title:
            return
        if kind == "SEARCH_LOGO" and not title:
            return
        topic_key = f"{topic_kind}:{title.lower()}"
        if topic_key in st.session_state.inflight_keys:
            return
        st.session_state.inflight_keys.add(topic_key)

        def worker():
            try:
                cands = websearch_agent("book" if kind == "SEARCH_COVER" else "tool", title, author)
                val = vision_validator_agent("book" if kind == "SEARCH_COVER" else "tool",
                                             expected_title=title,
                                             expected_author_or_name=(author or title),
                                             candidates=cands)
                best = val.get("best_index", -1)
                chosen = cands[best] if 0 <= best < len(cands) else {}
                RESULTS_QUEUE.put({
                    "topic_key": topic_key,
                    "topic_kind": topic_kind,
                    "title": title,
                    "image": chosen,
                    "candidates": cands,
                    "validator": val
                })
            finally:
                # inflight-flag freigeben passiert im Main-Thread beim Harvest
                pass

        threading.Thread(target=worker, daemon=True).start()
        return

    if kind == "SEARCH_PORTRAIT":
        name = action.get("name", "").strip()
        if not name:
            return
        topic_key = f"Person:{name.lower()}"
        if topic_key in st.session_state.inflight_keys:
            return
        st.session_state.inflight_keys.add(topic_key)

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
                    "topic_key": topic_key,
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

    # 3) Ask Director (visible question + optional ACTION lines)
    director_reply = director_ask_next()
    if director_reply:
        st.session_state.history.append({"role": "assistant", "content": director_reply})

        # 4) Parse and start actions (parallel, multi)
        actions = parse_actions(director_reply)
        for act in actions:
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
    topic_key = res.get("topic_key", "")
    if topic_key:
        st.session_state.inflight_keys.discard(topic_key)
    # map to topic by kind + title/name
    kind = res.get("topic_kind")
    title = res.get("title", "") or res.get("name", "")
    # find or create topic slot (based on kind)
    idx = None
    for i, t in enumerate(st.session_state.profile["topics"]):
        if t["name"] == kind:
            idx = i
            break
    if idx is None:
        continue
    t = st.session_state.profile["topics"][idx]
    t["media"]["chosen"] = res.get("image", {})
    t["media"]["candidates"] = res.get("candidates", [])
    t["media"]["validator"] = res.get("validator", {})
    # Markiere Research „done“, ohne den Interviewstatus zu überschreiben
    t["media"]["done"] = True
    # Ergebnis für die Liste (Anzeige)
    # vermeide Duplikate anhand image_url + kind
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

# Hinweis: laufende Jobs?
if st.session_state.inflight_keys:
    st.info("🔎 Background research running… you can keep chatting.")


# ─────────────────────────────────────────────
# Render chat history
# ─────────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ─────────────────────────────────────────────
# Finalizer trigger & rendering
# ─────────────────────────────────────────────
def all_research_done_for_answered_topics() -> bool:
    # Wir betrachten Topics, die mindestens eine Antwort haben – dort muss Research (falls angestoßen) fertig sein.
    for t in st.session_state.profile["topics"]:
        if t["answers"]:
            # wenn inflight zu diesem kind existiert → noch nicht fertig
            # (leichte Heuristik: prüfe 'done' Flag)
            if not t.get("media", {}).get("done"):
                return False
    return True

def finalizer_agent(profile: Dict[str, Any]) -> str:
    """
    Baut einen freundlichen, fokussierten Abschlussabsatz + kompakte Liste je Topic.
    Reine Sprache, keine JSON-Abhängigkeiten.
    """
    sys = (
        "You are a finalizer. Compose a concise, warm, professional summary (3–5 sentences) "
        "that presents the user positively without flattery. Then provide a compact list per item "
        "(Book/Podcast/Person/Tool) with a one-liner why it matters. Keep it crisp."
    )
    # Wir geben das Profil als einfacher Text weiter
    text_profile = []
    for t in profile["topics"]:
        if not t["answers"]:
            continue
        head = f"- {t['name']}:"
        facts = "; ".join([a.strip() for a in t["answers"] if a.strip()])
        chosen = t.get("media", {}).get("chosen", {})
        media_hint = chosen.get("image_url", "")
        text_profile.append(f"{head} {facts} (image: {media_hint})")
    user = "Interview facts:\n" + "\n".join(text_profile)

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

# Wenn Director FINALIZE_READY signalisiert hat UND alle Recherchen durch sind → finalisieren
if st.session_state.profile["finalizing"] and not st.session_state.inflight_keys and all_research_done_for_answered_topics():
    if not st.session_state.final_output:
        st.session_state.final_output = finalizer_agent(st.session_state.profile)

# Anzeige der Ergebnisse (Bilder + Kurztext), parallel zum Chat
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
            # kurzer, informeller Platzhaltertext; Finalizer liefert später schönen Text
            tk = res.get("topic_kind", "")
            title = res.get("title") or res.get("name") or ""
            st.markdown(f"**{tk}** — *{title}*")
            pg = res.get("image", {}).get("page_url", "")
            if pg:
                st.markdown(f"[Source]({pg})")

# Finaler Output (Text)
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
        # auch globale Queues leeren
        while not RESULTS_QUEUE.empty():
            try: RESULTS_QUEUE.get_nowait()
            except queue.Empty: break
        INFLIGHT_KEYS.clear()
        ensure_session()
        st.rerun()
with c2:
    if st.button("🔁 Refresh"):
        st.rerun()
