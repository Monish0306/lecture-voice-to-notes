import streamlit as st
import os
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import google.generativeai as genai

# ============
# Config
# ============
# Load Gemini API key
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", ""))

st.set_page_config(page_title="Lecture Voice → Notes (Gemini)", layout="wide")
st.title("🎙 Lecture Voice → Notes Generator (Gemini AI)")
st.markdown("""
Upload audio or video (mp3, wav, mp4...), I’ll transcribe and summarize into notes, flashcards, quiz.
""")

# =============
# Upload file
# =============
uploaded = st.file_uploader("Upload lecture audio or video", type=["mp3","wav","m4a","mp4","mov","mkv","avi"])
if uploaded:
    # Save temp
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        temp_input_path = tmp.name

    st.info("File uploaded, processing...")

    # If video, extract audio
    if suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
        try:
            video = VideoFileClip(temp_input_path)
            audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            video.audio.write_audiofile(audio_path, codec="pcm_s16le")
        except Exception as e:
            st.error(f"Error extracting audio from video: {e}")
            st.stop()
    else:
        audio_path = temp_input_path

    # Convert to clean WAV
    try:
        sound = AudioSegment.from_file(audio_path)
        sound = sound.set_channels(1)
        sound = sound.set_frame_rate(16000)
        converted = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sound.export(converted, format="wav")
        audio_path = converted
    except Exception as e:
        st.error(f"Audio conversion failed: {e}")
        st.stop()

    st.audio(audio_path)

    # Transcription
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as src:
            audio_data = recognizer.record(src)
        transcribed_text = recognizer.recognize_google(audio_data)
        st.success("✅ Transcription done")
    except Exception as e:
        st.error(f"Transcription error: {e}")
        transcribed_text = ""

    st.subheader("Transcribed Text")
    st.text_area("Transcript:", transcribed_text, height=200)

    # =============
    # Generate Notes
    # =============
    st.subheader("Generate Study Materials")
    user_model = st.text_input("Model to use (e.g. gemini-2.5-flash)", value="gemini-2.5-flash")
    if st.button("Generate Notes, Flashcards & Quiz"):
        if not transcribed_text.strip():
            st.warning("No transcript to process.")
        else:
            with st.spinner("Calling Gemini..."):
                try:
                    model = genai.GenerativeModel(user_model)
                    prompt = f"""
You are an academic assistant. From this transcript, produce:
1. A clear summary of the lecture.
2. 5 flashcards (Q → A).
3. 5 multiple-choice quiz questions + answers.

Transcript:
{transcribed_text}
"""
                    response = model.generate_content(prompt)
                    output = response.text

                    st.subheader("📚 Generated Notes & Quiz")
                    st.write(output)

                    st.download_button("Download", output, file_name="lecture_notes.txt")

                except Exception as e:
                    # If model not found, list available models
                    st.error(f"Error calling model: {e}")
                    try:
                        st.info("Trying to list available models...")
                        models = genai.list_models()
                        valid = [m.name for m in models if "generateContent" in m.supported_generation_methods]
                        st.write("Valid models are:", valid)
                    except Exception as e2:
                        st.error(f"Could not list models: {e2}")
