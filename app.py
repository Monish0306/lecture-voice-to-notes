import streamlit as st
import os
import tempfile
import speech_recognition as sr
import google.generativeai as genai
from pydub import AudioSegment
from moviepy.editor import VideoFileClip

# ========================================
# Configure Gemini API Key
# ========================================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ========================================
# Streamlit Page Setup
# ========================================
st.set_page_config(page_title="🎙️ Lecture Voice → Notes Generator", layout="wide")
st.title("🎧 Lecture Voice → Notes Generator (Gemini AI)")

st.markdown("""
This AI-powered app converts **lecture audio or video → text → summarized notes**  
and also creates **flashcards & quizzes** using **Google Gemini**.  
It automatically **filters background noise or music** and focuses on **spoken content**.
""")

# ========================================
# File Upload (Audio + Video)
# ========================================
uploaded_file = st.file_uploader(
    "📂 Upload Lecture Audio or Video File",
    type=["mp3", "wav", "m4a", "mp4", "mov", "mkv", "avi"]
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    input_suffix = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{input_suffix}") as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    st.info("📦 File uploaded successfully!")

    # ========================================
    # Extract Audio from Video (if applicable)
    # ========================================
    if input_suffix.lower() in ["mp4", "mov", "mkv", "avi"]:
        st.info("🎬 Extracting audio from video...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as extracted_audio:
            video = VideoFileClip(temp_input_path)
            audio = video.audio
            audio.write_audiofile(extracted_audio.name, codec='pcm_s16le')
            temp_audio_path = extracted_audio.name
    else:
        temp_audio_path = temp_input_path

    # ========================================
    # Convert to clean WAV format for recognition
    # ========================================
    st.info("🎧 Converting to compatible audio format (WAV)...")
    converted_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sound = AudioSegment.from_file(temp_audio_path)
    sound = sound.set_channels(1)  # mono
    sound = sound.set_frame_rate(16000)  # optimal for speech
    sound.export(converted_wav.name, format="wav")
    temp_audio_path = converted_wav.name

    st.audio(temp_audio_path)

    # ========================================
    # Speech-to-Text using SpeechRecognition
    # ========================================
    recognizer = sr.Recognizer()

    with sr.AudioFile(temp_audio_path) as source:
        st.info("🎤 Listening and transcribing (please wait)...")
        audio_data = recognizer.record(source)

        try:
            transcript_text = recognizer.recognize_google(audio_data)
            st.success("✅ Audio successfully transcribed!")
        except sr.UnknownValueError:
            st.error("❌ Could not understand the audio clearly.")
            transcript_text = ""
        except sr.RequestError:
            st.error("⚠️ Could not request results — check your internet connection.")
            transcript_text = ""

    # ========================================
    # Show Transcribed Text
    # ========================================
    st.subheader("📝 Transcribed Text")
    st.text_area("Transcript Output:", transcript_text, height=200)

    # ========================================
    # Generate Notes using Gemini
    # ========================================
    if st.button("✨ Generate Notes, Flashcards, and Quiz"):
        if not transcript_text.strip():
            st.warning("Please upload a valid lecture audio or video first.")
        else:
            model = genai.GenerativeModel("gemini-1.5-pro-latest")


            prompt = f"""
            You are an educational assistant. Based on this lecture transcript, generate:

            1️⃣ A **clear, well-structured summary** of the lecture.  
            2️⃣ 5 **Flashcards** (Question → Answer format).  
            3️⃣ 5 **Multiple-choice quiz questions** with 4 options each, and mark the correct answer.

            Remove irrelevant sounds or music from analysis.

            Transcript:
            {transcript_text}
            """

            with st.spinner("🧠 Generating study materials using Gemini..."):
                response = model.generate_content(prompt)

            st.subheader("📚 AI-Generated Notes, Flashcards, and Quiz")
            st.markdown(response.text)

            st.download_button(
                label="⬇️ Download Notes",
                data=response.text,
                file_name="lecture_notes.txt",
                mime="text/plain"
            )

            st.success("✅ All done!")
