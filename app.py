import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except Exception:
    TRANSLATOR_AVAILABLE = False

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="IntelliFAQ AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Session state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ----------------------------
# Language codes
# ----------------------------
lang_code = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
}

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: #f5f7fb;
        }

        .hero {
            text-align: center;
            padding: 0.25rem 0 1rem 0;
        }

        .hero h1 {
            margin: 0;
            color: #2563eb;
            font-size: 2.5rem;
            font-weight: 800;
        }

        .hero p {
            margin-top: 0.35rem;
            color: #667085;
            font-size: 1.02rem;
        }

        .sidebar-note {
            background: rgba(37, 99, 235, 0.08);
            padding: 0.8rem;
            border-radius: 12px;
            margin-top: 0.5rem;
            color: #111827;
        }

        .stButton > button {
            border-radius: 12px;
            height: 44px;
            font-weight: 600;
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🤖 IntelliFAQ AI Assistant</h1>
        <p>Smart AI knowledge assistant powered by Python • Streamlit • NLP</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Load FAQ data
# ----------------------------
faq_path = Path("faq.json")
if not faq_path.exists():
    st.error("faq.json not found. Please keep faq.json in the same folder as app.py.")
    st.stop()

try:
    faq_data = json.loads(faq_path.read_text(encoding="utf-8"))
except Exception as e:
    st.error(f"Could not read faq.json: {e}")
    st.stop()

if not isinstance(faq_data, list) or not faq_data:
    st.error("faq.json must contain a non-empty list of FAQ items.")
    st.stop()

questions = []
answers = []
for item in faq_data:
    if isinstance(item, dict):
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            questions.append(q)
            answers.append(a)

if not questions:
    st.error("No valid FAQ entries found. Each item must have both 'question' and 'answer'.")
    st.stop()

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def get_best_match(user_text: str):
    user_vec = vectorizer.transform([user_text])
    similarity = cosine_similarity(user_vec, question_vectors)[0]
    best_idx = int(similarity.argmax())
    best_score = float(similarity[best_idx])
    return best_idx, best_score

def get_related_questions(user_text: str, exclude_idx: int | None = None, top_n: int = 3):
    user_vec = vectorizer.transform([user_text])
    similarity = cosine_similarity(user_vec, question_vectors)[0]
    ranked = similarity.argsort()[::-1]

    related = []
    for idx in ranked:
        if exclude_idx is not None and idx == exclude_idx:
            continue
        if similarity[idx] <= 0:
            continue
        related.append(questions[idx])
        if len(related) >= top_n:
            break
    return related

def translate_answer(text: str, target_language: str) -> str:
    if target_language == "English":
        return text

    if not TRANSLATOR_AVAILABLE:
        return text

    try:
        return GoogleTranslator(
            source="auto",
            target=lang_code[target_language]
        ).translate(text)
    except Exception:
        return text

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🤖 IntelliFAQ AI")
st.sidebar.success("🟢 AI Online")
st.sidebar.metric("Questions Asked", st.session_state.questions_asked)

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Answer Language")
answer_language = st.sidebar.selectbox(
    "Choose answer language",
    list(lang_code.keys()),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Suggested Questions")

suggestions = [
    "What is Python?",
    "What is Artificial Intelligence?",
    "What is Java?",
    "What is Streamlit?",
    "What is GitHub?",
    "What is ChatGPT?",
]

for i, q in enumerate(suggestions):
    if st.sidebar.button(q, key=f"suggest_{i}"):
        st.session_state.pending_question = q
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="sidebar-note">
        <b>How it works</b><br>
        The chatbot compares your question with the FAQ list using TF-IDF + cosine similarity.
    </div>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
    st.session_state.questions_asked = 0
    st.session_state.pending_question = None
    st.rerun()

# ----------------------------
# Main view
# ----------------------------
if len(st.session_state.messages) == 0:
    st.info("👋 Welcome to IntelliFAQ AI! Ask about Python, AI, Java, Streamlit, GitHub and more.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None
else:
    prompt = st.chat_input("💬 Ask me anything...")

# ----------------------------
# Chat logic
# ----------------------------
if prompt:
    now_text = datetime.now().strftime("%d-%m-%Y %I:%M %p")

    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": now_text
    })
    st.session_state.questions_asked += 1

    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(now_text)

    best_idx, score = get_best_match(prompt)
    base_answer = answers[best_idx]
    related_questions = get_related_questions(prompt, exclude_idx=best_idx, top_n=3)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            time.sleep(0.4)

            if score >= 0.25:
                final_answer = translate_answer(base_answer, answer_language)

                with st.container(border=True):
                    st.markdown("### 🤖 AI Response")
                    st.write(final_answer)

                st.progress(score)
                st.caption(f"🎯 Confidence: {score * 100:.1f}%")
                st.caption(f"🕒 {now_text}")

                if answer_language != "English":
                    st.caption(f"🌐 Answer language: {answer_language}")

                if related_questions:
                    st.markdown("**📌 Related Questions**")
                    for rq in related_questions:
                        st.write(f"• {rq}")

            else:
                final_answer = (
                    "Sorry, I couldn't find an exact answer.\n\n"
                    "Try asking:\n"
                    "• What is Python?\n"
                    "• What is Artificial Intelligence?\n"
                    "• What is Java?\n"
                    "• What is Streamlit?\n"
                    "• What is GitHub?"
                )
                st.warning(final_answer)
                st.caption("ℹ️ Low confidence match")

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "timestamp": now_text
    })

    st.rerun()

# ----------------------------
# Download chat
# ----------------------------
if st.session_state.messages:
    chat_history = ""
    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "AI"
        chat_history += f"{role}: {msg['content']}\n\n"

    st.download_button(
        label="📥 Download Chat",
        data=chat_history,
        file_name="IntelliFAQ_Chat.txt",
        mime="text/plain"
    )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#667085;'>Developed by Balraj • CodeAlpha AI Internship</p>",
    unsafe_allow_html=True,
)