# app.py — Expert Card Creator (natural language agents, no JSON)
import os
import threading
import streamlit as st
from openai import OpenAI

# ======================
# 1) Setup
# ======================
st.set_page_config(page_title="Expert Card Creator", page_icon="📝", layout="centered")
st.title("📝 Expert Card Creator")

# OpenAI client & model config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.stop()  # Streamlit shows an error if missing; set in Secrets
client = OpenAI(api_key=OPENAI_API_KEY)

CHAT_MODEL = os.getenv("DTBR_CHAT_MODEL", "gpt-5")  # your default model

# ======================
# 2) Session State init
# ======================
if "profile" not in st.session_state:
    st.session_state.profile = {
        "topics": [
            {"name": "Book",    "status": "queued", "followups": 2, "answers": [], "research": None},
            {"name": "Podcast", "status": "queued", "followups": 2, "answers": [], "research": None},
            {"name": "Person",  "status": "queued", "followups": 1, "answers": [], "research": None},
            {"name": "Tool",    "status": "queued", "followups": 0, "answers": [], "research": None}
        ],
        "current_topic_index": 0
    }
    st.session_state.history = [
        {"role": "system", "content": "You are a helpful interviewer, creating an expert card from user answers. Keep it warm and focused. Avoid redundant confirmations."}
    ]
    st.session_state.history.append({
        "role": "assistant",
        "content": "Hi! I'll help you craft a tiny expert card. First, which book has helped you professionally? Please give the title (author optional)."
    })

# ======================
# 3) Helpers
# ======================
def handle_user_answer(user_input: str, state: dict) -> None:
    """Store answer and reduce follow-up count; advance topic when done."""
    topic = state["topics"][state["current_topic_index"]]
    topic["answers"].append(user_input)
    if topic["followups"] > 0:
        topic["followups"] -= 1
    if topic["followups"] <= 0:
        topic["status"] = "done"
        if state["current_topic_index"] + 1 < len(state["topics"]):
            state["current_topic_index"] += 1

def call_director(client: OpenAI, history: list[dict], state: dict) -> str:
    """Director agent: plain text; no JSON schema."""
    current_topic = state["topics"][state["current_topic_index"]]
    sys = (
        "You are the Director Agent for an expert card interview.\n"
        f"Current topic: {current_topic['name']}\n"
        f"Remaining follow-ups for this topic: {current_topic['followups']}\n"
        "- Ask ONE clear, fresh question about this topic that progresses the interview.\n"
        "- Avoid repeating earlier questions with the same meaning.\n"
        "- Max ~1–2 follow-ups per topic, then transition naturally to the next topic.\n"
        "- Do NOT ask for confirmation like 'Is that correct?'; only ask meaningful clarifications.\n"
        "- Keep answers concise and friendly."
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=history + [{"role": "system", "content": sys}],
            temperature=0.6,
        )
        # IMPORTANT: use attribute access, not dict indexing
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"(Director error: {e})"

def background_research(query: str, topic_name: str, state: dict) -> None:
    """Dummy non-blocking 'research' that simulates background work."""
    import time
    try:
        time.sleep(2)  # simulate IO
        result = f"(Background info for {topic_name}: found resources for '{query}')"
        # store result into the state object passed into the thread
        for t in state["topics"]:
            if t["name"] == topic_name and t.get("research") is None:
                t["research"] = result
                break
    except Exception:
        # swallow background exceptions to avoid noisy logs
        pass

# ======================
# 4) Input / Flow
# ======================
user_input = st.chat_input("Your answer…")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    handle_user_answer(user_input, st.session_state.profile)

    # Background research for some topics; this is non-blocking
    try:
        current_topic = st.session_state.profile["topics"][st.session_state.profile["current_topic_index"]]
        if current_topic["name"] in ["Podcast", "Person"]:
            threading.Thread(
                target=background_research,
                args=(user_input, current_topic["name"], st.session_state.profile),
                daemon=True
            ).start()
    except Exception:
        # if we're already past the last topic, skip background work
        pass

    # Ask director for the next question / smooth transition
    nxt = call_director(client, st.session_state.history, st.session_state.profile)
    if not nxt.strip():
        # minimal fallback prompt to keep chat flowing
        topic_name = st.session_state.profile["topics"][st.session_state.profile["current_topic_index"]]["name"]
        nxt = f"Thanks! One more on {topic_name.lower()}: what makes it stand out for you?"
    st.session_state.history.append({"role": "assistant", "content": nxt})

# ======================
# 5) Render chat
# ======================
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ======================
# 6) Optional: show background info as it arrives
# ======================
with st.expander("🔎 Background (as available)"):
    st.write(st.session_state.profile)
