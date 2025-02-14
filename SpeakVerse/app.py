import streamlit as st
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
from io import BytesIO
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

# Load Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

st.title("Multilingual Audio Translator 🎙️🌍")

# Class for processing audio stream
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_data = BytesIO()

    def recv(self, frame):
        audio_frame = frame.to_ndarray().tobytes()
        self.audio_data.write(audio_frame)
        return frame

# Language mapping for gTTS
language_mapping = {
    "Urdu": "ur",
    "Arabic": "ar",
    "French": "fr",
    "Spanish": "es"
}

# Choose Input Method
input_method = st.radio("Select Input Method:", ("Upload Audio File", "Record Audio"))

audio_path = None  # Initialize the audio path

if input_method == "Upload Audio File":
    # Upload Audio File
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])
    if uploaded_file is not None:
        audio_path = "uploaded_audio.wav"
        audio_segment = AudioSegment.from_file(uploaded_file)
        audio_segment.export(audio_path, format="wav")
        st.audio(audio_path, format="audio/wav")

elif input_method == "Record Audio":
    # Record Audio
    st.header("Record Audio")
    webrtc_ctx = webrtc_streamer(key="audio", audio_processor_factory=AudioProcessor)

    if webrtc_ctx.state.playing:
        st.warning("Recording... Stop to proceed.")
    elif webrtc_ctx.audio_processor:
        st.success("Recording finished! Processing...")
        # Save recorded audio
        audio_path = "recorded_audio.wav"
        recorded_audio = webrtc_ctx.audio_processor.audio_data
        recorded_audio.seek(0)
        audio_segment = AudioSegment.from_raw(recorded_audio, sample_width=2, frame_rate=44100, channels=1)
        audio_segment.export(audio_path, format="wav")
        # Display the recorded audio
        st.audio(audio_path, format="audio/wav")

# Proceed if there's an audio file to process
if audio_path:
    # User selects the target language
    target_language = st.selectbox("Select Target Language:", ("Urdu", "Arabic", "French", "Spanish"))
    gtts_language = language_mapping[target_language]

    # Transcribe Audio using Whisper Model
    st.info("Transcribing audio...")
    result = model.transcribe(audio_path)
    text = result["text"]
    st.write("Transcribed Text:", text)

    # Translate the text using GoogleTranslator
    st.info("Translating text...")
    translated_text = GoogleTranslator(source='auto', target=gtts_language).translate(text)
    st.write(f"Translated Text ({target_language}):", translated_text)

    # Convert Translated Text to Speech using gTTS
    st.info("Converting translated text to speech...")
    tts = gTTS(translated_text, lang=gtts_language)
    translated_audio_path = "translated_audio.mp3"
    tts.save(translated_audio_path)
    st.audio(translated_audio_path, format="audio/mp3")

    # Provide download button for translated audio
    with open(translated_audio_path, "rb") as f:
        st.download_button("Download Translated Audio", data=f, file_name="translated_audio.mp3", mime="audio/mp3")
    
    # Cleanup temporary files
    os.remove(audio_path)
    os.remove(translated_audio_path)
