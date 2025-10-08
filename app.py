import streamlit as st
import os
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import google.generativeai as genai

# =====================================
# 🌟 PAGE CONFIGURATION
# =====================================
st.set_page_config(
    page_title="🎧 Lecture → Notes AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================
# 🎨 CUSTOM CSS FOR STYLING
# =====================================
st.markdown("""
<style>
/* Main app background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1a1a1a 0%, #2c3e50 100%);
    color: white;
}

/* Headings */
h1, h2, h3 {
    color: #00bfff;
    text-shadow: 1px 1px 2px #00000070;
}

/* Cards and sections */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Buttons */
div.stButton > button:first-child {
    background-color: #00bfff;
    color: white;
    font-weight: 600;
    border-radius: 12px;
    border: none;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #0080ff;
    transform: scale(1.05);
}

/* Text areas */
textarea {
    border-radius: 10px !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background-color: #00bfff;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}

/* Divider */
hr {
    border: 1px solid #00bfff33;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #111 !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# 🧠 HEADER
# =====================================
st.markdown("""
# 🎧 **Lecture Voice → Notes Generator (Gemini AI)**
Transform your lecture **audio or video** into:
- 📝 Summarized notes  
- 💡 Flashcards  
- 🧩 Quizzes  

Powered by **Google Gemini AI** ⚡  
""")

# =====================================
# 🔑 GEMINI CONFIG
# =====================================
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")))

# =====================================
# 📂 FILE UPLOAD
# =====================================
st.divider()
st.header("📂 Upload Your Lecture")
st.caption("🎵 MP3, WAV, M4A, MP4, MOV, MKV, AVI (max 200MB)")

uploaded = st.file_uploader("📤 Choose a file", type=["mp3", "wav", "m4a", "mp4", "mov", "mkv", "avi"])

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        input_path = tmp.name

    st.success(f"✅ Uploaded: `{uploaded.name}`")

    # =====================================
    # 🎬 CONVERT TO AUDIO
    # =====================================
    with st.spinner("🎧 Extracting & converting audio..."):
        try:
            if suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
                video = VideoFileClip(input_path)
                audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                video.audio.write_audiofile(audio_path, codec="pcm_s16le", verbose=False, logger=None)
            else:
                audio_path = input_path

            sound = AudioSegment.from_file(audio_path)
            sound = sound.set_channels(1).set_frame_rate(16000)
            processed_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            sound.export(processed_path, format="wav")
        except Exception as e:
            st.error(f"❌ Audio conversion failed: {e}")
            st.stop()

    st.audio(processed_path, format="audio/wav")
    st.success("🎵 Audio ready for transcription!")

    # =====================================
    # 🗣 TRANSCRIBE AUDIO
    # =====================================
    recognizer = sr.Recognizer()
    with st.spinner("📝 Transcribing lecture... (please wait)"):
        try:
            with sr.AudioFile(processed_path) as src:
                audio_data = recognizer.record(src)
            transcribed_text = recognizer.recognize_google(audio_data)
            st.success("✅ Transcription complete!")
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")
            transcribed_text = ""

    st.subheader("📜 Transcript")
    st.text_area("Lecture Transcript:", transcribed_text, height=250)

    # =====================================
    # 🤖 GEMINI AI GENERATION
    # =====================================
    st.divider()
    st.header("🧠 Generate Study Materials")
    model_name = st.text_input("⚙️ Gemini Model:", value="gemini-2.0-flash")

    if st.button("✨ Generate Notes, Flashcards & Quizzes"):
        if not transcribed_text.strip():
            st.warning("⚠️ Please transcribe the lecture before generating content.")
            st.stop()

        with st.spinner("🤖 Generating with Gemini..."):
            try:
                model = genai.GenerativeModel(model_name)
                prompt = f"""
You are an academic assistant AI.

Based on the following lecture transcript, create:

1️⃣ **Concise Lecture Notes (with bullet points and emojis if relevant)**  
2️⃣ **5 Flashcards (Q&A format)**  
3️⃣ **5 Multiple-Choice Quiz Questions (4 options each, highlight correct answer)**  

Make it engaging, cleanly formatted, and helpful for students.

🎤 Transcript:
{transcribed_text}
"""
                response = model.generate_content(prompt)
                result = response.text.strip()
                st.success("✅ Content generated successfully!")
                st.subheader("📚 Study Materials")
                st.markdown(result)

                st.download_button(
                    "📥 Download Results",
                    data=result,
                    file_name="lecture_notes.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"❌ Gemini API Error: {e}")
                try:
                    models = genai.list_models()
                    valid = [m.name for m in models if "generateContent" in m.supported_generation_methods]
                    st.info("✅ Available models:")
                    st.write(valid)
                except Exception as e2:
                    st.error(f"⚠️ Could not list models: {e2}")

# =====================================
# ⚙️ FOOTER
# =====================================
st.divider()
st.markdown("""
<center>
💡 *Developed by Monish Valiveti*  
Built with ❤️ using **Streamlit + Google Gemini AI**  
🌐 [GitHub Repository](https://github.com/) | ☁️ [Deployed on Streamlit Cloud](https://share.streamlit.io)
</center>
""", unsafe_allow_html=True)
