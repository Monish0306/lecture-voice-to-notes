import streamlit as st
import os
import tempfile
import json
import re
import time
import numpy as np
import soundfile as sf
import librosa
from datetime import datetime
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import requests

# Try to import Faster Whisper (BEST for production)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    st.error("Faster-Whisper not installed. Run: pip install faster-whisper")

# Try to import sounddevice (optional for recording feature)
try:
    import sounddevice as sd
    from scipy.io.wavfile import write
    RECORDING_AVAILABLE = True
except (ImportError, OSError):
    RECORDING_AVAILABLE = False

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
# 🔐 PASSWORD VALIDATION
# =====================================
def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter (A-Z)"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter (a-z)"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number (0-9)"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special symbol (!@#$%^&*)"
    return True, "Password is strong!"

# =====================================
# 🔐 LOGIN SYSTEM
# =====================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_data" not in st.session_state:
    st.session_state.user_data = {}

if not st.session_state.logged_in:
    st.markdown("# 🔐 Welcome to Lecture Notes AI")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
        
        with tab1:
            st.markdown("### Login to your account")
            login_username = st.text_input("👤 Username", placeholder="Enter your username", key="login_user")
            login_password = st.text_input("🔒 Password", type="password", placeholder="Enter password", key="login_pass")
            
            if st.button("🚀 Login", use_container_width=True):
                if login_username and login_password:
                    if login_username in st.session_state.user_data:
                        st.session_state.logged_in = True
                        st.session_state.username = login_username
                        st.success(f"Welcome back, {login_username}!")
                        st.rerun()
                    else:
                        st.error("❌ User not found. Please sign up first.")
                else:
                    st.error("❌ Please enter username and password")
        
        with tab2:
            st.markdown("### Create a new account")
            signup_username = st.text_input("👤 Choose Username", placeholder="Enter username", key="signup_user")
            signup_password = st.text_input("🔒 Choose Password", type="password", placeholder="Enter password", key="signup_pass")
            
            if signup_password:
                is_valid, message = validate_password(signup_password)
                if is_valid:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
            
            if st.button("📝 Create Account", use_container_width=True):
                if signup_username and signup_password:
                    is_valid, message = validate_password(signup_password)
                    if not is_valid:
                        st.error(f"❌ {message}")
                    elif signup_username in st.session_state.user_data:
                        st.error("❌ Username already exists. Please choose another.")
                    else:
                        st.session_state.logged_in = True
                        st.session_state.username = signup_username
                        st.session_state.user_data[signup_username] = {
                            "password": signup_password,
                            "history": [],
                            "total_quizzes": 0,
                            "total_score": 0
                        }
                        st.success(f"✅ Account created! Welcome {signup_username}!")
                        st.rerun()
                else:
                    st.error("❌ Please fill in all fields")
    st.stop()

# =====================================
# 🧠 HEADER WITH USER INFO
# =====================================
col_header1, col_header2 = st.columns([4, 1])
with col_header1:
    st.markdown("""
    # 🎧 Lecture → Notes Generator (AI Powered)
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

GEMINI_MODEL = "gemini-2.0-flash-exp"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"

# Initialize session state
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "current_view" not in st.session_state:
    st.session_state.current_view = "home"
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = None
if "summarized_notes" not in st.session_state:
    st.session_state.summarized_notes = ""
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

# =====================================
# 🎙 RECORD OR UPLOAD
# =====================================
uploaded = None

if RECORDING_AVAILABLE:
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
    st.sidebar.info("🎤 Recording feature not available in cloud deployment")

if not uploaded:
    uploaded = st.file_uploader("📂 Upload Audio or Video",
                                type=["mp3", "wav", "m4a", "mp4", "mov", "mkv", "avi"])

# =====================================
# 🤖 LOAD WHISPER MODEL
# =====================================
@st.cache_resource
def load_whisper_model():
    """Load Faster Whisper model"""
    if not FASTER_WHISPER_AVAILABLE:
        return None
    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# =====================================
# 🎤 AUDIO ENHANCEMENT
# =====================================
def enhance_quiet_audio(audio_data, sample_rate):
    """Enhance very quiet audio without distortion"""
    max_amplitude = np.max(np.abs(audio_data))
    
    if max_amplitude < 0.1:  # Very quiet audio
        audio_data = audio_data / max_amplitude * 0.7
        audio_data = librosa.effects.preemphasis(audio_data)
    
    return audio_data

# =====================================
# 🗣 TRANSCRIPTION
# =====================================
def transcribe_audio_bytes(model, audio_bytes: bytes):
    """Transcribe audio with enhanced accuracy"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile_path = tmpfile.name
            
            # Read audio
            audio_data, current_sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
            
            # Resample if needed
            if current_sr != 16000:
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=1)
                audio_data = librosa.resample(audio_data, orig_sr=current_sr, target_sr=16000)
            
            # Ensure mono
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Enhance quiet audio
            audio_data = enhance_quiet_audio(audio_data, 16000)
            
            # Write processed audio
            sf.write(tmpfile_path, audio_data, 16000)
        
        # Transcribe
        segments, info = model.transcribe(
            tmpfile_path,
            beam_size=5,
            language="en",
            vad_filter=True,
            word_timestamps=True
        )
        
        full_transcript = " ".join(segment.text for segment in segments)
        
        # Cleanup
        if os.path.exists(tmpfile_path):
            os.remove(tmpfile_path)
        
        return full_transcript.strip(), info.language
        
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None, None

if uploaded and not st.session_state.transcribed_text:
    overall_start_time = time.time()
    
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        input_path = tmp.name
    
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_text = st.empty()
    
    status_text.info("🎧 **Step 1/3:** Extracting and converting audio...")
    time_text.text(f"📊 File size: {file_size_mb:.2f} MB")
    
    try:
        if suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
            progress_bar.progress(10)
            video = VideoFileClip(input_path)
            audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            video.audio.write_audiofile(audio_path, codec="pcm_s16le", verbose=False, logger=None)
            progress_bar.progress(25)
        else:
            audio_path = input_path
            progress_bar.progress(25)

        sound = AudioSegment.from_file(audio_path)
        
        status_text.info("🎵 **Step 2/3:** Enhancing audio quality...")
        
        sound = sound.normalize(headroom=0.1)
        progress_bar.progress(35)
        
        sound = sound.compress_dynamic_range(threshold=-25.0, ratio=6.0, attack=3.0, release=40.0)
        progress_bar.progress(45)
        
        sound = sound.high_pass_filter(80)
        sound = sound.low_pass_filter(8000)
        progress_bar.progress(55)
        
        sound = sound + 12
        sound = sound.set_channels(1).set_frame_rate(16000)
        progress_bar.progress(65)
        
        processed_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sound.export(processed_path, format="wav")
        progress_bar.progress(70)
        
        conversion_time = time.time() - overall_start_time
        time_text.success(f"✅ Audio processing completed in {conversion_time:.1f}s")
        
    except Exception as e:
        status_text.error(f"❌ Audio conversion failed: {e}")
        progress_bar.empty()
        time_text.empty()
        st.stop()

    # Transcription
    audio_duration = len(sound) / 1000.0
    
    status_text.info(f"📝 **Step 3/3:** Transcribing {audio_duration:.0f}s of audio...")
    
    model = load_whisper_model()
    
    if model:
        try:
            progress_bar.progress(75)
            status_text.info("🤖 Loading Faster Whisper AI...")
            
            with open(processed_path, "rb") as f:
                audio_bytes = f.read()
            
            progress_bar.progress(80)
            transcribe_start = time.time()
            
            transcript, language = transcribe_audio_bytes(model, audio_bytes)
            
            if transcript:
                transcribe_time = time.time() - transcribe_start
                progress_bar.progress(100)
                
                st.session_state.transcribed_text = transcript
                
                total_time = time.time() - overall_start_time
                total_mins = int(total_time // 60)
                total_secs = int(total_time % 60)
                
                status_text.success(f"✅ Transcription complete!")
                if total_mins > 0:
                    time_text.success(f"⏱️ **Total: {total_mins}min {total_secs}s** | Words: {len(transcript.split())} | Language: {language}")
                else:
                    time_text.success(f"⏱️ **Total: {total_secs}s** | Words: {len(transcript.split())} | Language: {language}")
                
                st.success("🎉 **Perfect transcription achieved!**")
                
                time.sleep(2)
                progress_bar.empty()
                
        except Exception as e:
            status_text.error(f"❌ Transcription failed: {e}")
            time_text.empty()
            progress_bar.empty()
    else:
        st.error("❌ Whisper model not available")

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
def generate_study_materials(transcript):
    """Generate notes, flashcards, and quiz using Gemini"""
    try:
        prompt = f"""Based on this lecture transcript, create a comprehensive study package in JSON format:

{{
  "notes": "Detailed summary with key concepts (400-500 words, use markdown formatting)",
  "flashcards": [
    {{"q": "Question about key concept", "a": "Clear, concise answer"}},
    // Generate 8-10 flashcards covering main topics
  ],
  "quiz": [
    {{
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": "Correct option text"
    }}
    // Generate exactly 5 questions
  ]
}}

Transcript: {transcript}"""

        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        response = requests.post(API_URL, headers={'Content-Type': 'application/json'}, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
            
            # Clean JSON
            text = text.replace('```json', '').replace('```', '').strip()
            data = json.loads(text)
            return data
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Generation error: {e}")
        return None

if st.session_state.transcribed_text:
    st.divider()
    st.header("🧠 Generate Study Materials")

    if st.button("✨ Generate Notes, Flashcards & Quiz"):
        with st.spinner("🤖 Generating with Gemini AI..."):
            data = generate_study_materials(st.session_state.transcribed_text)
            
            if data:
                st.session_state.quiz_data = data
                st.session_state.summarized_notes = data.get("notes", "")
                st.session_state.flashcards = data.get("flashcards", [])
                st.session_state.current_view = "notes"
                st.session_state.quiz_submitted = False
                st.session_state.quiz_results = None
                st.success("✅ Study materials generated!")
                st.rerun()

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
            st.session_state.quiz_submitted = False
            st.session_state.quiz_results = None
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
    st.markdown(st.session_state.summarized_notes)
    
    st.download_button(
        "📥 Download Notes",
        st.session_state.summarized_notes,
        file_name="lecture_notes.md",
        mime="text/markdown"
    )

# =====================================
# 💡 FLASHCARDS VIEW
# =====================================
if st.session_state.quiz_data and st.session_state.current_view == "flashcards":
    st.markdown("## 💡 Flashcards")
    for i, fc in enumerate(st.session_state.flashcards, 1):
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
    
    if not st.session_state.quiz_submitted:
        with st.form(key="quiz_submission_form", clear_on_submit=False):
            user_answers = []
            
            for i, q in enumerate(quiz_questions):
                st.markdown(f"### Q{i+1}. {q['question']}")
                
                selected = st.radio(
                    f"Select your answer:",
                    options=q["options"],
                    key=f"q_{i}",
                    index=None,
                    label_visibility="collapsed"
                )
                user_answers.append(selected)
                st.markdown("---")
            
            all_answered = all(ans is not None for ans in user_answers)
            
            if not all_answered:
                st.warning("⚠️ Please answer all questions before submitting!")
            
            submitted = st.form_submit_button("✅ Submit Quiz", use_container_width=True, type="primary")
            
            if submitted:
                if all_answered:
                    score = sum(1 for i, q in enumerate(quiz_questions) if user_answers[i] == q["answer"])
                    total = len(quiz_questions)
                    percent = (score / total) * 100
                    
                    st.session_state.quiz_results = {
                        "score": score,
                        "total": total,
                        "percent": percent,
                        "user_answers": user_answers,
                        "questions": quiz_questions
                    }
                    st.session_state.quiz_submitted = True
                    
                    user_history = st.session_state.user_data[st.session_state.username]["history"]
                    user_history.append({
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "notes": st.session_state.summarized_notes,
                        "flashcards": st.session_state.flashcards,
                        "quiz_score": f"{score}/{total}",
                        "percentage": percent
                    })
                    
                    st.session_state.user_data[st.session_state.username]["total_quizzes"] += 1
                    st.session_state.user_data[st.session_state.username]["total_score"] += score
                    
                    st.rerun()
                else:
                    st.error("❌ Please answer all questions!")
    
    if st.session_state.quiz_submitted and st.session_state.quiz_results:
        results = st.session_state.quiz_results
        
        st.divider()
        st.subheader("📊 Quiz Results")
        
        for i, q in enumerate(results["questions"]):
            user_ans = results["user_answers"][i]
            correct_ans = q["answer"]
            is_correct = user_ans == correct_ans
            
            st.markdown(f"**Q{i+1}. {q['question']}**")
            st.markdown(f"Your answer: **{user_ans}**")
            
            if is_correct:
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Wrong! Correct answer: **{correct_ans}**")
            st.markdown("---")
        
        st.markdown("### 🏆 Final Score")
        st.success(f"**{results['score']}/{results['total']} ({results['percent']:.1f}%)**")

        if results['percent'] < 70:
            st.warning("😕 Try again after reviewing the notes.")
        elif results['percent'] < 90:
            st.info("👍 Good work! You're improving.")
        else:
            st.balloons()
            st.success("🎉 Excellent! Perfect understanding!")
        
        if st.button("🔄 Retake Quiz", use_container_width=True):
            st.session_state.quiz_submitted = False
            st.session_state.quiz_results = None
            st.rerun()

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