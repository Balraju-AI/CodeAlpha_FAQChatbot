import json
import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI FAQ Assistant",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top, #111827 0%, #0f172a 45%, #020617 100%);
            color: #F8FAFC;
        }

        .hero {
            background: linear-gradient(90deg, #2563EB 0%, #7C3AED 100%);
            padding: 28px 30px;
            border-radius: 22px;
            color: white;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.28);
            border: 1px solid rgba(255,255,255,0.14);
            margin-bottom: 18px;
        }

        .hero h1 {
            margin: 0;
            font-size: 42px;
            line-height: 1.1;
        }

        .hero p {
            margin: 10px 0 0 0;
            font-size: 17px;
            opacity: 0.95;
        }

        .user-bubble {
            background: linear-gradient(135deg, #2563EB, #1D4ED8);
            color: white;
            padding: 16px 18px;
            border-radius: 18px 18px 4px 18px;
            margin: 10px 0 8px 18%;
            box-shadow: 0 10px 18px rgba(37,99,235,.22);
            border: 1px solid rgba(255,255,255,0.12);
        }

        .bot-bubble {
            background: rgba(255,255,255,0.10);
            color: white;
            padding: 16px 18px;
            border-radius: 18px 18px 18px 4px;
            margin: 8px 18% 12px 0;
            box-shadow: 0 10px 18px rgba(0,0,0,.18);
            border: 1px solid rgba(255,255,255,0.12);
        }

        .footer {
            text-align: center;
            margin-top: 24px;
            opacity: 0.75;
            font-size: 14px;
        }

        .stButton > button {
            background: linear-gradient(90deg, #16a34a, #22c55e);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 0.7rem 1rem;
            font-weight: 700;
            box-shadow: 0 8px 18px rgba(34,197,94,0.20);
        }

        .stButton > button:hover {
            filter: brightness(1.04);
        }

        .stTextInput input {
            border-radius: 14px !important;
            background: rgba(255,255,255,0.05) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

if "latest" not in st.session_state:
    st.session_state.latest = None

try:
    with open("faq.json", "r", encoding="utf-8") as f:
        faq_data = json.load(f)
except Exception as e:
    st.error(f"Could not load faq.json: {e}")
    st.stop()

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

with st.sidebar:
    st.markdown("## 🤖 AI FAQ Assistant")
    st.caption("CodeAlpha AI Internship Project")
    st.markdown("---")
    st.markdown("### 📌 About")
    st.write("A professional FAQ chatbot built with Python, Streamlit, and NLP.")
    st.write("TF-IDF + Cosine Similarity for question matching.")
    st.markdown("---")
    st.markdown("### ⚙️ Actions")

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.latest = None
        st.rerun()

    st.download_button(
        "📥 Download Chat History",
        data="\n\n".join(
            f"You: {q}\nAI: {a}\nConfidence: {c:.2f}%"
            for q, a, c in st.session_state.history
        ) if st.session_state.history else "No chat history yet.",
        file_name="chat_history.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.markdown("---")
    st.success(f"✅ FAQs Loaded: {len(faq_data)}")
    st.info(f"💬 Questions Asked: {len(st.session_state.history)}")

    avg_conf = (
        sum(c for _, _, c in st.session_state.history) / len(st.session_state.history)
        if st.session_state.history else 0.0
    )
    st.info(f"🎯 Avg Confidence: {avg_conf:.2f}%")

st.markdown(
    """
    <div class="hero">
        <h1>🤖 AI FAQ Assistant</h1>
        <p>Ask a question and get an intelligent answer instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
col1.metric("FAQs Loaded", len(faq_data))
col2.metric("Questions Asked", len(st.session_state.history))
avg_conf_main = (
    sum(c for _, _, c in st.session_state.history) / len(st.session_state.history)
    if st.session_state.history else 0.0
)
col3.metric("Avg Confidence", f"{avg_conf_main:.2f}%")

st.markdown("### 💬 Ask anything")
st.caption("Example: What is Machine Learning?")

user_question = st.text_input(
    "Ask a question",
    placeholder="Type your question here...",
    label_visibility="collapsed",
)

ask_clicked = st.button("🚀 Ask AI", use_container_width=True)

if ask_clicked:
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🤖 AI is thinking..."):
            time.sleep(0.6)

        user_vector = vectorizer.transform([user_question])
        similarity = cosine_similarity(user_vector, question_vectors)

        index = similarity.argmax()
        score = float(similarity[0][index])
        confidence = score * 100
        matched_question = questions[index]

        if score > 0.25:
            answer = answers[index]
        else:
            answer = "Sorry, I couldn't find an answer."

        st.session_state.latest = {
            "question": user_question,
            "answer": answer,
            "confidence": confidence,
            "matched_question": matched_question,
            "score": score,
        }

        st.session_state.history.append((user_question, answer, confidence))

if st.session_state.latest:
    result = st.session_state.latest

    st.markdown("### 🤖 Latest Answer")
    st.markdown(f"**Matched Question:** {result['matched_question']}")
    st.markdown(f"**Similarity Score:** {result['score']:.2f}")

    st.markdown(
        f"""
        <div class="user-bubble">
            <b>👤 You</b><br>
            {result['question']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="bot-bubble">
            <b>🤖 AI Assistant</b><br>
            {result['answer']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(result["confidence"] / 100, 1.0))
    st.caption(f"🎯 Confidence: {result['confidence']:.2f}%")

    speak_col, _ = st.columns([1, 3])
    with speak_col:
        if st.button("🔊 Speak Answer", use_container_width=True):
            try:
                tts = gTTS(text=result["answer"], lang="en")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    audio_path = tmp.name

                tts.save(audio_path)
                audio_bytes = Path(audio_path).read_bytes()
                st.audio(audio_bytes, format="audio/mp3")

                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            except Exception as e:
                st.warning(f"Voice output unavailable: {e}")

if st.session_state.history:
    st.markdown("### 💬 Chat History")

    for q, a, confidence in reversed(st.session_state.history):
        st.markdown(
            f"""
            <div class="user-bubble">
                <b>👤 You</b><br>
                {q}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="bot-bubble">
                <b>🤖 AI Assistant</b><br>
                {a}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(min(confidence / 100, 1.0))
        st.caption(f"🎯 Confidence: {confidence:.2f}%")

st.markdown(
    "<div class='footer'>Developed by <b>Balraj</b> | CodeAlpha AI Internship</div>",
    unsafe_allow_html=True,
)