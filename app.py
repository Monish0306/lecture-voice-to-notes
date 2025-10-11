import streamlit as st
import os
import tempfile
import json
import re
import time
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import google.generativeai as genai
from datetime import datetime

# Try to import Whisper for better transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Try AssemblyAI as alternative
try:
    import assemblyai as aai
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False

# Always import speech_recognition as fallback
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# Try to import sounddevice (optional for recording feature)
try:
    import sounddevice as sd
    from scipy.io.wavfile import write
    import numpy as np
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
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False

# Simple login (you can enhance this with database)
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
                    # Check if user exists (simple check - in production use proper authentication)
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
            
            # Real-time password validation
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
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = None

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
# 🗣 TRANSCRIPTION WITH WHISPER (BEST ACCURACY)
# =====================================
def clean_transcription(text):
    """Clean transcription but keep all meaningful content"""
    # Remove excessive repetitions
    words = text.split()
    cleaned_words = []
    prev_word = ""
    repeat_count = 0
    
    for word in words:
        if word.lower() == prev_word.lower():
            repeat_count += 1
            if repeat_count < 2:  # Allow one repetition
                cleaned_words.append(word)
        else:
            cleaned_words.append(word)
            repeat_count = 0
        prev_word = word
    
    return ' '.join(cleaned_words).strip()

if uploaded and not st.session_state.transcribed_text:
    # Start overall timer
    overall_start_time = time.time()
    
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        input_path = tmp.name
    
    # Get file size for progress estimation
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    
    # Audio conversion phase
    conversion_start = time.time()
    
    # Create a progress bar and status area
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_text = st.empty()
    
    status_text.info("🎧 **Step 1/3:** Extracting and converting audio...")
    time_text.text(f"📊 File size: {file_size_mb:.2f} MB | Estimated time: {file_size_mb * 2:.0f}s")
    
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
        
        status_text.info("🎵 **Step 2/3:** Enhancing audio quality for better accuracy...")
        
        # ENHANCED: Professional-grade audio enhancement
        # Normalize audio volume
        sound = sound.normalize()
        progress_bar.progress(35)
        
        # Apply compression to make quiet parts louder
        sound = sound.compress_dynamic_range(threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
        progress_bar.progress(45)
        
        # Remove noise while preserving speech
        sound = sound.high_pass_filter(80)   # Remove very low frequency noise
        sound = sound.low_pass_filter(8000)  # Remove very high frequency noise
        progress_bar.progress(55)
        
        # Boost overall volume
        sound = sound + 12  # Increase by 12dB
        
        # Convert to optimal format for transcription
        sound = sound.set_channels(1).set_frame_rate(16000)
        progress_bar.progress(65)
        
        processed_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sound.export(processed_path, format="wav")
        progress_bar.progress(70)
        
        conversion_time = time.time() - conversion_start
        time_text.success(f"✅ Audio processing completed in {conversion_time:.1f}s")
        
    except Exception as e:
        status_text.error(f"❌ Audio conversion failed: {e}")
        progress_bar.empty()
        time_text.empty()
        st.stop()

    # Transcription phase
    transcription_start = time.time()
    audio_duration = len(sound) / 1000.0  # Duration in seconds
    
    status_text.info(f"📝 **Step 3/3:** Transcribing {audio_duration:.0f}s of audio...")
    
    # Estimate transcription time
    if WHISPER_AVAILABLE:
        estimated_time = audio_duration * 0.3  # Whisper is ~0.3x realtime
        time_text.text(f"⏱️ Using Whisper AI | Estimated time: {estimated_time:.0f}s | Audio: {audio_duration:.0f}s")
    elif ASSEMBLYAI_AVAILABLE:
        estimated_time = audio_duration * 0.5  # AssemblyAI is ~0.5x realtime
        time_text.text(f"⏱️ Using AssemblyAI | Estimated time: {estimated_time:.0f}s | Audio: {audio_duration:.0f}s")
    else:
        estimated_time = audio_duration * 0.8  # Google SR is ~0.8x realtime
        time_text.text(f"⏱️ Using Google Speech Recognition | Estimated time: {estimated_time:.0f}s | Audio: {audio_duration:.0f}s")

    # Use Whisper if available (BEST accuracy - works offline)
    if WHISPER_AVAILABLE:
        try:
            progress_bar.progress(75)
            status_text.info("🤖 Loading Whisper AI model...")
            
            # Load Whisper model
            model_load_start = time.time()
            model = whisper.load_model("base")
            model_load_time = time.time() - model_load_start
            
            progress_bar.progress(80)
            status_text.info(f"🎯 Transcribing with Whisper AI (Model loaded in {model_load_time:.1f}s)...")
            
            # Transcribe with optimal settings
            transcribe_start = time.time()
            result = model.transcribe(
                processed_path,
                language="en",
                fp16=False,
                verbose=False,
                temperature=0.0,
                best_of=5,
                beam_size=5,
                word_timestamps=True,
                condition_on_previous_text=True
            )
            transcribe_time = time.time() - transcribe_start
            
            progress_bar.progress(95)
            
            raw_text = result["text"]
            cleaned_text = clean_transcription(raw_text)
            
            st.session_state.transcribed_text = cleaned_text if cleaned_text else raw_text
            
            progress_bar.progress(100)
            total_time = time.time() - overall_start_time
            
            status_text.success(f"✅ Transcription complete with Whisper AI!")
            time_text.success(f"⏱️ **Total time: {total_time:.1f}s** | Transcription: {transcribe_time:.1f}s | Words: {len(raw_text.split())} | Speed: {audio_duration/transcribe_time:.1f}x realtime")
            
            # Clear progress after 3 seconds
            time.sleep(2)
            progress_bar.empty()
            
        except Exception as e:
            status_text.error(f"❌ Whisper failed: {e}")
            time_text.info("Trying alternative method...")
            progress_bar.progress(70)
    
    # Try AssemblyAI if Whisper not available (Cloud-based, very accurate)
    elif ASSEMBLYAI_AVAILABLE:
        ASSEMBLYAI_KEY = st.secrets.get("ASSEMBLYAI_API_KEY", os.getenv("ASSEMBLYAI_API_KEY", ""))
        if ASSEMBLYAI_KEY:
            try:
                progress_bar.progress(75)
                aai.settings.api_key = ASSEMBLYAI_KEY
                transcriber = aai.Transcriber()
                
                config = aai.TranscriptionConfig(
                    speech_model=aai.SpeechModel.best,
                    language_detection=True,
                    punctuate=True,
                    format_text=True
                )
                
                progress_bar.progress(80)
                status_text.info("☁️ Uploading to AssemblyAI servers...")
                
                upload_start = time.time()
                transcript = transcriber.transcribe(processed_path, config=config)
                
                # Poll for completion
                while transcript.status not in [aai.TranscriptStatus.completed, aai.TranscriptStatus.error]:
                    time.sleep(1)
                    elapsed = time.time() - upload_start
                    time_text.text(f"⏱️ Processing... {elapsed:.0f}s elapsed")
                    progress_bar.progress(min(95, 80 + int(elapsed * 2)))
                
                if transcript.status == aai.TranscriptStatus.completed:
                    raw_text = transcript.text
                    cleaned_text = clean_transcription(raw_text)
                    st.session_state.transcribed_text = cleaned_text if cleaned_text else raw_text
                    
                    progress_bar.progress(100)
                    total_time = time.time() - overall_start_time
                    
                    status_text.success("✅ Professional transcription complete!")
                    time_text.success(f"⏱️ **Total time: {total_time:.1f}s** | Words: {len(raw_text.split())}")
                    
                    time.sleep(2)
                    progress_bar.empty()
                else:
                    status_text.error(f"❌ Transcription failed: {transcript.error}")
                    
            except Exception as e:
                status_text.error(f"❌ AssemblyAI failed: {e}")
        else:
            status_text.warning("⚠️ AssemblyAI API key not found. Using fallback method.")
    
    # Fallback to speech_recognition if nothing else available
    if not st.session_state.transcribed_text and SR_AVAILABLE:
        recognizer = sr.Recognizer()
        
        # ENHANCED: Optimized recognition settings
        recognizer.energy_threshold = 50
        recognizer.dynamic_energy_threshold = True
        recognizer.dynamic_energy_adjustment_damping = 0.05
        recognizer.dynamic_energy_ratio = 1.1
        recognizer.pause_threshold = 1.2
        recognizer.phrase_threshold = 0.05
        recognizer.non_speaking_duration = 0.5
        
        try:
            progress_bar.progress(75)
            status_text.info("🎤 Analyzing audio...")
            
            with sr.AudioFile(processed_path) as src:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(src, duration=2.0)
                progress_bar.progress(85)
                
                status_text.info("📡 Transcribing with Google Speech Recognition...")
                audio_data = recognizer.record(src)
            
            progress_bar.progress(90)
            transcribe_start = time.time()
            
            # Use Google Speech Recognition
            raw_text = recognizer.recognize_google(
                audio_data,
                language="en-US",
                show_all=False
            )
            
            transcribe_time = time.time() - transcribe_start
            progress_bar.progress(95)
            
            # Minimal cleaning - keep everything
            cleaned_text = clean_transcription(raw_text)
            
            if len(cleaned_text.strip()) > 0:
                st.session_state.transcribed_text = cleaned_text
            else:
                st.session_state.transcribed_text = raw_text
            
            progress_bar.progress(100)
            total_time = time.time() - overall_start_time
            
            status_text.success("✅ Transcription complete!")
            time_text.success(f"⏱️ **Total time: {total_time:.1f}s** | Transcription: {transcribe_time:.1f}s | Words: {len(cleaned_text.split())}")
            st.info("💡 **Tip:** Install Whisper AI for higher accuracy: `pip install openai-whisper`")
            
            time.sleep(2)
            progress_bar.empty()
                
        except sr.UnknownValueError:
            progress_bar.empty()
            status_text.error("❌ Could not understand audio. Please ensure:\n- Audio is clear and audible\n- Minimal background noise\n- Speaker is close to microphone")
            time_text.info("💡 **Tip:** For best results, install Whisper AI: `pip install openai-whisper torch`")
            st.session_state.transcribed_text = ""
        except sr.RequestError as e:
            progress_bar.empty()
            status_text.error(f"❌ Transcription service error: {e}")
            time_text.empty()
            st.session_state.transcribed_text = ""
        except Exception as e:
            progress_bar.empty()
            status_text.error(f"❌ Transcription failed: {e}")
            time_text.empty()
            st.session_state.transcribed_text = ""
    
    elif not st.session_state.transcribed_text:
        progress_bar.empty()
        status_text.error("❌ No transcription service available. Please install required packages.")
        time_text.info("**Run:** `pip install SpeechRecognition` or `pip install openai-whisper`")

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
                  "notes": "Concise and clear summary in markdown (max 400 words)",
                  "flashcards": [
                    {{"q": "Question1", "a": "Answer1"}},
                    {{"q": "Question2", "a": "Answer2"}}
                  ],
                  "quiz": [
                    {{"question": "Question text", "options": ["Option A", "Option B", "Option C", "Option D"], "answer": "Option A"}}
                  ]
                }}
                Generate **exactly 5 quiz questions**, each with 4 distinct options (A–D).
                Make the notes comprehensive and detailed, extracting all key concepts from the lecture.
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
                st.session_state.quiz_submitted = False
                st.session_state.quiz_results = None
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
# 🧩 QUIZ VIEW - FIXED VERSION
# =====================================
if st.session_state.quiz_data and st.session_state.current_view == "quiz":
    st.markdown("## 🧩 Interactive Quiz")
    
    quiz_questions = st.session_state.quiz_data["quiz"][:5]
    
    # Show quiz form only if not submitted
    if not st.session_state.quiz_submitted:
        # Use a unique key for the form
        with st.form(key="quiz_submission_form", clear_on_submit=False):
            user_answers = []
            
            for i, q in enumerate(quiz_questions):
                st.markdown(f"### Q{i+1}. {q['question']}")
                
                # Use index=-1 to show no selection initially (not supported by radio)
                # So we use index=None by not setting it, but radio needs an index
                # Solution: Use index=None is not valid, so we'll use selectbox instead
                selected = st.radio(
                    f"Select your answer:",
                    options=q["options"],
                    key=f"q_{i}",
                    index=None,  # No pre-selection in newer Streamlit versions
                    label_visibility="collapsed"
                )
                user_answers.append(selected)
                st.markdown("---")
            
            # Check if all answered
            all_answered = all(ans is not None for ans in user_answers)
            
            if not all_answered:
                st.warning("⚠️ Please answer all questions before submitting!")
            
            submitted = st.form_submit_button("✅ Submit Quiz", use_container_width=True, type="primary")
            
            if submitted:
                if all_answered:
                    # Calculate score
                    score = sum(1 for i, q in enumerate(quiz_questions) if user_answers[i] == q["answer"])
                    total = len(quiz_questions)
                    percent = (score / total) * 100
                    
                    # Store results in session state
                    st.session_state.quiz_results = {
                        "score": score,
                        "total": total,
                        "percent": percent,
                        "user_answers": user_answers,
                        "questions": quiz_questions
                    }
                    st.session_state.quiz_submitted = True
                    
                    # Save to history
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
                    
                    st.rerun()
                else:
                    st.error("❌ Please answer all questions!")
    
    # Show results if submitted
    if st.session_state.quiz_submitted and st.session_state.quiz_results:
        results = st.session_state.quiz_results
        
        st.divider()
        st.subheader("📊 Quiz Results")
        
        # Show detailed results
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
        
        # Final score
        st.markdown("### 🏆 Final Score")
        st.success(f"**{results['score']}/{results['total']} ({results['percent']:.1f}%)**")

        if results['percent'] < 70:
            st.warning("😕 Try again after reviewing the notes.")
        elif results['percent'] < 90:
            st.info("👍 Good work! You're improving.")
        else:
            st.balloons()
            st.success("🎉 Excellent! Perfect understanding!")
        
        # Retake button
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