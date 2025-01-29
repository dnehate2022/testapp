import streamlit as st
from openai import OpenAI
import tempfile

# 🌟 Load API Key from Streamlit Secrets (For Deployment)
api_key = st.secrets["OPENAI_API_KEY"]

# 🌍 Initialize OpenAI Client
client = OpenAI(api_key=api_key)

# 🌟 Streamlit Page Config
st.set_page_config(page_title="Whisper Transcription & AI Summary", layout="centered")
st.title("🎙️ Audio Transcription, Summary & Q&A")

# ✅ Initialize session state for transcription & summary
if "transcriptions" not in st.session_state:
    st.session_state["transcriptions"] = {}
if "summaries" not in st.session_state:
    st.session_state["summaries"] = {}

# 📤 Upload Audio File
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

# 📝 Function to Transcribe Audio
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcription

# ✍️ Function to Generate Summary
def generate_summary(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Summarize the following transcript concisely."},
                  {"role": "user", "content": text}]
    )
    return response.choices[0].message.content.strip()

# ❓ Function for Q&A Based on Summary
def ask_question(question, summary):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You answer questions based on the provided summary."},
                  {"role": "user", "content": f"Summary: {summary}\n\nQuestion: {question}"}]
    )
    return response.choices[0].message.content.strip()

# 🚀 Handle File Upload & Processing
if uploaded_file is not None:
    file_key = uploaded_file.name  # Unique key for each file

    if file_key not in st.session_state["transcriptions"]:
        st.info("🔄 Processing and transcribing... Please wait.")

        # 🔄 Save uploaded file temporarily (Auto-Deletes After Use)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file.flush()  # Ensure data is written to file

            # 🎙️ Transcribe Audio
            transcription = transcribe_audio(temp_file.name)

        # ✅ Store Transcription
        st.session_state["transcriptions"][file_key] = transcription

        # 🔍 Generate Summary
        summary = generate_summary(transcription)
        st.session_state["summaries"][file_key] = summary

    # ✅ Display Transcribed Text
    st.success(f"✅ Transcription Completed for: {file_key}")
    st.text_area("📜 Transcribed Text:", st.session_state["transcriptions"][file_key], height=300)

    # Display the Summary
    st.subheader("📝 Summary of the Transcription")
    st.write(st.session_state["summaries"][file_key])

    # ❓ Q&A Section
    st.subheader("🤖 Ask a Question About the Summary")
    question = st.text_input("Enter your question:")

    if question:
        answer = ask_question(question, st.session_state["summaries"][file_key])
        st.write("💡 **Answer:**", answer)

else:
    st.info("📂 Please upload an audio file to start transcription.")
