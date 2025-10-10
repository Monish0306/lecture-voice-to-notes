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

/* Navigation buttons styling */
.nav-button {
    display: inline-block;
    padding: 12px 24px;
    margin: 5px;
    background: linear-gradient(135deg, #00bfff, #0080ff);
    color: white;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
    text-decoration: none;
    cursor: pointer;
    transition: all 0.3s ease;
}
.nav-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 191, 255, 0.4);
}
.nav-button.active {
    background: linear-gradient(135deg, #00ff88, #00cc66);
}
</style>
""", unsafe_allow_html=True)

# =====================================
# 🔐 LOGIN SYSTEM
# =====================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_data" not in st.session_state:
    st.session_state.user_data = {}

# Simple login (you can enhance this with database)
if not st.session_state.logged_in:
    st.markdown("# 🔐 Welcome to Lecture Notes AI")
    st.markdown("### Please login to continue")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🚀 Login", use_container_width=True):
                if username and password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    # Load user data if exists
                    if username not in st.session_state.user_data:
                        st.session_state.user_data[username] = {
                            "history": [],
                            "total_quizzes": 0,
                            "total_score": 0
                        }
                    st.rerun()
                else:
                    st.error("Please enter username and password")
        
        with col_b:
            if st.button("📝 Sign Up", use_container_width=True):
                if username and password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_data[username] = {
                        "history": [],
                        "total_quizzes": 0,
                        "total_score": 0
                    }
                    st.success(f"Welcome {username}! Account created.")
                    st.rerun()
    st.stop()

# =====================================
# 🧠 HEADER WITH USER INFO
# =====================================
col_header1, col_header2 = st.columns([4, 1])
with col_header1:
    st.markdown("""
    # 🎧 Lecture → Notes Generator (Gemini AI)
    Transform lecture audio/video into study materials
    """)
with col_header2:
    st.markdown(f"### 👤 {st.session_state.username}")
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# =====================================
# 🔑 GEMINI CONFIG
# =====================================
API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if not API_KEY:
    st.error("❌ Missing Gemini API Key. Set GEMINI_API_KEY in secrets or environment.")
    st.stop()

genai.configure(api_key=API_KEY)

# Initialize session state
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "current_view" not in st.session_state:
    st.session_state.current_view = "home"

# =====================================
# 🎙 RECORD OR UPLOAD
# =====================================
st.sidebar.header("🎤 Record Audio")
record_duration = st.sidebar.slider("Recording Duration (seconds)", 5, 60, 10)

uploaded = None
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
    uploaded = st.file_uploader("📂 Upload Audio or Video",
                                type=["mp3", "wav", "m4a", "mp4", "mov", "mkv", "avi"])

# =====================================
# 🗣 TRANSCRIPTION
# =====================================
if uploaded and not st.session_state.transcribed_text:
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
            st.session_state.transcribed_text = recognizer.recognize_google(audio_data)
            st.success("✅ Transcription complete!")
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")
            st.session_state.transcribed_text = ""

# =====================================
# 📜 DISPLAY TRANSCRIPT
# =====================================
if st.session_state.transcribed_text.strip():
    st.subheader("📜 Transcribed Notes")

    st.sidebar.header("🧰 Text Appearance")
    font_size = st.sidebar.slider("Font Size", 12, 30, 18)
    font_choice = st.sidebar.selectbox("Font Style",
                                       ["Arial", "Georgia", "Courier New", "Verdana", "Times New Roman", 
                                        "Trebuchet MS", "Comic Sans MS", "Impact", "Lucida Console", 
                                        "Palatino Linotype", "Garamond", "Bookman", "Tahoma"], index=0)

    st.markdown(
        f"""
        <div style='background-color:#1e1e1e;padding:15px;border-radius:12px;
        font-family:{font_choice};font-size:{font_size}px;color:#fff;line-height:1.6;
        border:1px solid #00bfff55;max-height:400px;overflow-y:auto;'>
        {st.session_state.transcribed_text.replace('\n', '<br>')}
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================
# 🤖 GENERATE NOTES, FLASHCARDS, QUIZ
# =====================================
if st.session_state.transcribed_text:
    st.divider()
    st.header("🧠 Generate Study Materials")
    model_name = "gemini-2.0-flash"

    if st.button("✨ Generate Notes, Flashcards & Quiz"):
        with st.spinner("🤖 Generating using Gemini..."):
            try:
                model = genai.GenerativeModel(model_name)
                prompt = f"""
                You are an academic assistant AI. Based on the lecture transcript below,
                create a JSON with these exact keys:
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
                Transcript: {st.session_state.transcribed_text}
                """
                response = model.generate_content(prompt)
                raw_text = response.text.strip()
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                raw_text = raw_text.replace("json", "").strip()
                data = json.loads(raw_text)

                st.session_state.quiz_data = data
                st.session_state.current_view = "notes"
                st.success("✅ Study materials generated successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Gemini API Error: {e}")

# =====================================
# 🎯 NAVIGATION BUTTONS
# =====================================
if st.session_state.quiz_data:
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📝 Notes", use_container_width=True):
            st.session_state.current_view = "notes"
            st.rerun()
    
    with col2:
        if st.button("💡 Flashcards", use_container_width=True):
            st.session_state.current_view = "flashcards"
            st.rerun()
    
    with col3:
        if st.button("🧩 Quiz", use_container_width=True):
            st.session_state.current_view = "quiz"
            st.rerun()
    
    with col4:
        if st.button("📊 History", use_container_width=True):
            st.session_state.current_view = "history"
            st.rerun()

# =====================================
# 📝 NOTES VIEW
# =====================================
if st.session_state.quiz_data and st.session_state.current_view == "notes":
    st.markdown("## 📝 Study Notes")
    st.markdown(st.session_state.quiz_data.get("notes", ""))

# =====================================
# 💡 FLASHCARDS VIEW
# =====================================
if st.session_state.quiz_data and st.session_state.current_view == "flashcards":
    st.markdown("## 💡 Flashcards")
    for i, fc in enumerate(st.session_state.quiz_data.get("flashcards", []), 1):
        with st.expander(f"🃏 Flashcard {i}: {fc['q']}", expanded=False):
            st.markdown(f"### Question:")
            st.info(fc['q'])
            st.markdown(f"### Answer:")
            st.success(fc['a'])

# =====================================
# 🧩 QUIZ VIEW
# =====================================
if st.session_state.quiz_data and st.session_state.current_view == "quiz":
    st.markdown("## 🧩 Interactive Quiz")
    
    quiz_questions = st.session_state.quiz_data["quiz"][:5]
    
    with st.form("quiz_form"):
        user_answers = []
        
        for i, q in enumerate(quiz_questions):
            st.markdown(f"### Q{i+1}. {q['question']}")
            selected = st.radio(
                f"Select answer for Q{i+1}:",
                options=q["options"],
                key=f"quiz_q{i}",
                label_visibility="collapsed"
            )
            user_answers.append(selected)
            st.markdown("---")
        
        submit_button = st.form_submit_button("✅ Submit Quiz", use_container_width=True)
        
        if submit_button:
            score = 0
            total = len(quiz_questions)
            
            for i, q in enumerate(quiz_questions):
                if user_answers[i] == q["answer"]:
                    score += 1
            
            st.divider()
            st.subheader("📊 Quiz Results")
            
            for i, q in enumerate(quiz_questions):
                is_correct = user_answers[i] == q["answer"]
                
                st.markdown(f"**Q{i+1}. {q['question']}**")
                st.markdown(f"Your answer: **{user_answers[i]}**")
                
                if is_correct:
                    st.success(f"✅ Correct!")
                else:
                    st.error(f"❌ Wrong! Correct answer: **{q['answer']}**")
                
                st.markdown("---")
            
            percent = (score / total) * 100
            st.markdown("### 🏆 Final Score")
            st.success(f"**{score}/{total} ({percent:.1f}%)**")

            if percent < 70:
                st.warning("😕 Try again after reviewing the notes.")
            elif percent < 90:
                st.info("👍 Good work! You're improving.")
            else:
                st.balloons()
                st.success("🎉 Excellent! Perfect understanding!")

            # Save to user's history
            user_history = st.session_state.user_data[st.session_state.username]["history"]
            user_history.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "notes": st.session_state.quiz_data.get("notes", ""),
                "flashcards": st.session_state.quiz_data.get("flashcards", []),
                "quiz_score": f"{score}/{total}",
                "percentage": percent
            })
            
            st.session_state.user_data[st.session_state.username]["total_quizzes"] += 1
            st.session_state.user_data[st.session_state.username]["total_score"] += score

# =====================================
# 📚 HISTORY VIEW
# =====================================
if st.session_state.current_view == "history":
    st.markdown("## 📚 Your Learning History")
    
    user_history = st.session_state.user_data.get(st.session_state.username, {}).get("history", [])
    total_quizzes = st.session_state.user_data.get(st.session_state.username, {}).get("total_quizzes", 0)
    total_score = st.session_state.user_data.get(st.session_state.username, {}).get("total_score", 0)
    
    if total_quizzes > 0:
        avg_score = (total_score / (total_quizzes * 5)) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Quizzes", total_quizzes)
        with col2:
            st.metric("⭐ Total Score", f"{total_score}/{total_quizzes * 5}")
        with col3:
            st.metric("📈 Average Score", f"{avg_score:.1f}%")
        
        st.divider()
    
    if not user_history:
        st.info("No previous notes yet. Start by uploading a lecture!")
    else:
        for i, h in enumerate(reversed(user_history), 1):
            with st.expander(f"📅 Session {i} - {h['time']} - Score: {h['quiz_score']} ({h.get('percentage', 0):.1f}%)"):
                st.markdown("### 📝 Notes")
                st.markdown(h["notes"])
                
                st.markdown("### 💡 Flashcards")
                for fc in h["flashcards"]:
                    st.markdown(f"**Q:** {fc['q']}")
                    st.markdown(f"**A:** {fc['a']}")
                    st.markdown("---")
                
                st.markdown(f"### 🏆 Quiz Score: {h['quiz_score']}")

# =====================================
# 📚 SIDEBAR QUICK STATS
# =====================================
st.sidebar.divider()
st.sidebar.header("📊 Your Stats")
user_stats = st.session_state.user_data.get(st.session_state.username, {})
st.sidebar.metric("Total Quizzes", user_stats.get("total_quizzes", 0))
if user_stats.get("total_quizzes", 0) > 0:
    avg = (user_stats.get("total_score", 0) / (user_stats.get("total_quizzes", 0) * 5)) * 100
    st.sidebar.metric("Average Score", f"{avg:.1f}%")