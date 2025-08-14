# app.py
import streamlit as st
import threading
from openai import OpenAI

# ======================
# 1. Setup
# ======================
st.set_page_config(page_title="Expert Card Creator", page_icon="📝", layout="centered")
st.title("📝 Expert Card Creator")

# OpenAI Client
client = OpenAI()

# ======================
# 2. Initialisierung State
# ======================
if "profile" not in st.session_state:
    st.session_state.profile = {
        "topics": [
            {"name": "Book", "status": "queued", "followups": 2, "answers": [], "research": None},
            {"name": "Podcast", "status": "queued", "followups": 2, "answers": [], "research": None},
            {"name": "Person", "status": "queued", "followups": 1, "answers": [], "research": None},
            {"name": "Tool", "status": "queued", "followups": 0, "answers": [], "research": None}
        ],
        "current_topic_index": 0
    }
    st.session_state.history = [
        {"role": "system", "content": "You are a helpful interviewer, creating an expert card from user answers."}
    ]
    # Erste Frage sofort setzen
    st.session_state.history.append({
        "role": "assistant",
        "content": "Hi! I'll help you craft a tiny expert card. First, which book has helped you professionally? Please give the title (author optional)."
    })


# ======================
# 3. Hilfsfunktionen
# ======================
def handle_user_answer(user_input, state):
    """Speichert Antwort und reduziert Follow-up Counter."""
    topic = state["topics"][state["current_topic_index"]]
    topic["answers"].append(user_input)
    if topic["followups"] > 0:
        topic["followups"] -= 1
    if topic["followups"] <= 0:
        topic["status"] = "done"
        if state["current_topic_index"] + 1 < len(state["topics"]):
            state["current_topic_index"] += 1


def call_director(client, history, state):
    """Fragt den Director Agent in natürlicher Sprache."""
    current_topic = state["topics"][state["current_topic_index"]]
    system_prompt = f"""
    You are the Director Agent. 
    Current topic: {current_topic['name']}
    You have {current_topic['followups']} follow-up questions left for this topic.
    Ask one clear and fresh question related to this topic. 
    Avoid repeating the same question.
    If follow-ups are done, naturally transition to the next topic.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history + [{"role": "system", "content": system_prompt}],
        temperature=0.6
    )
    return response.choices[0].message["content"]


def background_research(query, topic_name, state):
    """Simuliert eine Websuche im Hintergrund."""
    import time
    time.sleep(2)  # Dummy-Verzögerung
    result = f"(Background info for {topic_name}: Found related resources for '{query}')"
    # Speichert Ergebnis im State
    for topic in state["topics"]:
        if topic["name"] == topic_name:
            topic["research"] = result
            break


# ======================
# 4. Eingabe & Ablauf
# ======================
user_input = st.chat_input("Your answer...")

if user_input:
    # User-Eingabe speichern
    st.session_state.history.append({"role": "user", "content": user_input})
    handle_user_answer(user_input, st.session_state.profile)

    # Falls Thema "Podcast" oder "Person" → Hintergrundsuche starten
    current_topic = st.session_state.profile["topics"][st.session_state.profile["current_topic_index"]]
    if current_topic["name"] in ["Podcast", "Person"]:
        threading.Thread(
            target=background_research,
            args=(user_input, current_topic["name"], st.session_state.profile),
            daemon=True
        ).start()

    # Director für nächste Frage fragen
    next_question = call_director(client, st.session_state.history, st.session_state.profile)
    st.session_state.history.append({"role": "assistant", "content": next_question})


# ======================
# 5. Chat-Verlauf anzeigen
# ======================
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ======================
# 6. Debug-Bereich (optional)
# ======================
with st.expander("🔍 Debug Info"):
    st.write(st.session_state.profile)
