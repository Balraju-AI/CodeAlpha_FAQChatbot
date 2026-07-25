import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import os


# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 180px;
    font-size: 16px;
    font-weight: bold;
}

.stTextInput>div>div>input {
    border-radius: 10px;
    font-size: 17px;
}

h1 {
    color: #1E3A8A;
    text-align: center;
}

.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("<h1>🤖 AI FAQ Chatbot</h1>", unsafe_allow_html=True)
    st.write("CodeAlpha AI Internship Project")

    st.markdown("---")

    st.info(
        """
        **Features**
        - AI-based FAQ Search
        - Chat History
        - Clear Chat
        - Fast Responses
        """
    )
    # ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("🤖 AI FAQ Chatbot")
    st.markdown("### CodeAlpha AI Internship")

    st.markdown("---")

    st.write("### Features")
    st.write("✅ AI-based FAQ Search")
    st.write("✅ Chat History")
    st.write("✅ Clear Chat")
    st.write("✅ Fast Responses")

    st.markdown("---")
    st.info("Developed by Balraj")

st.title("🤖 AI FAQ Chatbot")

st.markdown("""
Welcome to the **AI FAQ Chatbot**.

Type your question below and the chatbot will find the most relevant answer from the FAQ database.
""")
if "history" not in st.session_state:
    st.session_state.history = []
st.markdown("---")
st.subheader("📊 Statistics")
st.write(f"Total Questions: {len(st.session_state.history)}")
successful = sum(
    1 for _, ans, _ in st.session_state.history
    if ans != "Sorry, I couldn't find an answer."
)

st.write(f"Successful Answers: {successful}")
if st.session_state.history:
    avg = sum(c for _, _, c in st.session_state.history) / len(st.session_state.history)
    st.write(f"Average Confidence: {avg:.2f}%")

# ----------------------------
# Load FAQ Data
# ----------------------------
with open("faq.json", "r", encoding="utf-8") as file:
    faq_data = json.load(file)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# ----------------------------
# Build AI Model
# ----------------------------
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# ----------------------------
# Chat History
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# User Input
# ----------------------------
user_question = st.text_input("💬 Ask a question")

# ----------------------------
# Get Answer
# ----------------------------
if st.button("Get Answer"):

    if user_question.strip() == "":
        st.warning("Please enter a question.")

    else:
        user_vector = vectorizer.transform([user_question])

        similarity = cosine_similarity(user_vector, question_vectors)

        index = similarity.argmax()
        score = similarity[0][index]
        confidence = score * 100

        st.write("Matched Question:", questions[index])
        st.write("Score:", score)

        if score > 0.25:
            answer = answers[index]
        else:
            answer = "Sorry, I couldn't find an answer."

        st.session_state.history.append((user_question, answer, confidence))
# ----------------------------
# Display Chat History
# ----------------------------
if st.session_state.history:
    st.subheader("💬 Chat History")

    for question, answer, confidence in reversed(st.session_state.history):
     st.markdown(f"**👤 You:** {question}")
     st.success(answer)
     tts = gTTS(text=answer, lang="en")
    tts.save("answer.mp3")

    audio_file = open("answer.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")
    st.info(f"🎯 Confidence: {confidence:.2f}%")

        # ----------------------------
# Download Chat History
# ----------------------------
if st.session_state.history:

    chat_text = ""

    for question, answer, confidence in st.session_state.history:
     chat_text += f"You: {question}\n"
     chat_text += f"Bot: {answer}\n\n"

    st.download_button(
        label="📥 Download Chat History",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )

# ----------------------------
# Clear Chat
# ----------------------------
if st.button("🗑 Clear Chat"):
    st.session_state.history = []
    st.rerun()

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    "<div class='footer'>Developed by <b>Balraj</b> | CodeAlpha AI Internship</div>",
    unsafe_allow_html=True
)