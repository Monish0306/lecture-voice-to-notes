import streamlit as st
import os
import tempfile
import json
import speech_recognition as sr
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import google.generativeai as genai
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from datetime import datetime

# =====================================
# 🌟 PAGE CONFIGURATION
# =====================================
st.set_page_config(
    page_title="🎧 Lecture → Notes AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# 🎨 CUSTOM CSS
# =====================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1a1a1a 0%, #2c3e50 100%);
    color: white;
}
h1, h2, h3 { color: #00bfff; text-shadow: 1px 1px 2px #00000070; }
div.stButton > button:first-child {
    background-color: #00bfff;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    border: none;
}
div.stButton > button:hover {
    background-color: #0080ff;
    transform: scale(1.05);
}
[data-testid="stSidebar"] { background-color: #111 !important; }
</style>
""", unsafe_allow_html=True)

# =====================================
# 🧠 HEADER
# =====================================
st.markdown("""
# 🎧 Lecture → Notes Generator (Gemini AI)
Transform lecture audio/video into:
- 📝 Summarized Notes  
- 💡 Flashcards  
- 🧩 Interactive Quiz  
""")

# =====================================
# 🔑 GEMINI CONFIG
# =====================================
API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if not API_KEY:
    st.error("❌ Missing Gemini API Key. Set GEMINI_API_KEY in secrets or environment.")
    st.stop()

genai.configure(api_key=API_KEY)

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================
# 🎙 RECORD AUDIO
# =====================================
st.sidebar.header("🎤 Record Audio")
record_duration = st.sidebar.slider("Recording Duration (seconds)", 5, 60, 10)
if st.sidebar.button("⏺ Start Recording"):
    st.info("🎙 Recording...")
    fs = 16000
    recording = sd.rec(int(record_duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_audio.name, fs, recording)
    st.sidebar.success(f"✅ Recorded {record_duration} seconds of audio!")
    uploaded = open(temp_audio.name, "rb")
else:
    uploaded = st.file_uploader("📂 Upload Audio or Video", type=["mp3", "wav", "m4a", "mp4", "mov", "mkv", "avi"])

# =====================================
# 🗣 TRANSCRIPTION
# =====================================
transcribed_text = ""
if uploaded:
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        input_path = tmp.name

    with st.spinner("🎧 Extracting and converting audio..."):
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

    recognizer = sr.Recognizer()
    with st.spinner("📝 Transcribing lecture..."):
        try:
            with sr.AudioFile(processed_path) as src:
                audio_data = recognizer.record(src)
            transcribed_text = recognizer.recognize_google(audio_data)
            st.success("✅ Transcription complete!")
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")
            transcribed_text = ""

# =====================================
# 📜 DISPLAY TRANSCRIPT
# =====================================
if transcribed_text.strip():
    st.subheader("📜 Transcribed Notes")

    st.sidebar.header("🧰 Text Appearance")
    font_size = st.sidebar.slider("Font Size", 12, 30, 18)
    font_choice = st.sidebar.selectbox("Font Style", ["Arial", "Georgia", "Courier New", "Verdana", "Times New Roman"], index=0)

    st.markdown(
        f"""
        <div style='background-color:#1e1e1e;padding:15px;border-radius:12px;
        font-family:{font_choice};font-size:{font_size}px;color:#fff;line-height:1.6;
        border:1px solid #00bfff55;max-height:400px;overflow-y:auto;'>
        {transcribed_text.replace('\n', '<br>')}
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================
# 🤖 GENERATE NOTES, FLASHCARDS, QUIZ
# =====================================
if transcribed_text:
    st.divider()
    st.header("🧠 Generate Study Materials")
    model_name = "gemini-2.0-flash"

    if st.button("✨ Generate Notes, Flashcards & Quiz"):
        with st.spinner("🤖 Generating using Gemini..."):
            try:
                model = genai.GenerativeModel(model_name)
                prompt = f"""
You are an academic assistant AI.
Based on the lecture transcript below, create a JSON with these exact keys:
{{
  "notes": "Concise and clear summary in markdown (max 300 words)",
  "flashcards": [
    {{"q": "Question1", "a": "Answer1"}},
    {{"q": "Question2", "a": "Answer2"}}
  ],
  "quiz": [
    {{"question": "Question text", "options": ["Option A", "Option B", "Option C", "Option D"], "answer": "Option A"}}
  ]
}}
Generate **exactly 5 quiz questions**, each with 4 distinct options (A–D).
Transcript:
{transcribed_text}
"""
                response = model.generate_content(prompt)
                raw_text = response.text.strip()
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                raw_text = raw_text.replace("json", "").strip()
                data = json.loads(raw_text)

                # ----------------------------
                # 📝 NOTES
                # ----------------------------
                st.subheader("📝 Notes")
                st.markdown(data["notes"])

                # ----------------------------
                # 💡 FLASHCARDS
                # ----------------------------
                st.subheader("💡 Flashcards")
                for fc in data["flashcards"]:
                    st.markdown(f"**Q:** {fc['q']}  \n💬 **A:** {fc['a']}")

                # ----------------------------
                # 🧩 QUIZ
                # ----------------------------
                st.subheader("🧩 Quiz (5 Questions)")
                user_answers = {}
                for i, q in enumerate(data["quiz"][:5]):  # ensure only 5 questions
                    st.markdown(f"**Q{i+1}. {q['question']}**")
                    selected = st.radio("Choose your answer:", q["options"], key=f"q_{i}")
                    user_answers[i] = (selected == q["answer"])

                if st.button("✅ Submit Quiz"):
                    score = sum(1 for v in user_answers.values() if v)
                    total = len(user_answers)
                    percent = (score / total) * 100
                    st.success(f"🏆 Score: {score}/{total} ({percent:.1f}%)")
                    if percent < 70:
                        st.warning("😕 Try again after reviewing the notes.")
                    elif percent < 90:
                        st.info("👍 Good work! You’re improving.")
                    else:
                        st.balloons()
                        st.success("🎉 Excellent! Perfect understanding!")

                    st.session_state.history.append({
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "notes": data["notes"],
                        "flashcards": data["flashcards"],
                        "quiz_score": f"{score}/{total}"
                    })

            except Exception as e:
                st.error(f"❌ Gemini API Error: {e}")

# =====================================
# 📚 HISTORY
# =====================================
st.sidebar.header("📚 History")
if not st.session_state.history:
    st.sidebar.info("No previous notes yet.")
else:
    for h in st.session_state.history[::-1]:
        with st.sidebar.expander(f"🗓 {h['time']}"):
            st.markdown(h["notes"])
            st.markdown(f"**Last Quiz Score:** {h['quiz_score']}")
