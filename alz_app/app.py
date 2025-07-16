import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import os
import tempfile

# Page settings
st.set_page_config(page_title="Alzheimer's Early Detection", layout="centered")

# App title
st.title(" Alzheimer’s Early Detection App")
st.write("Upload a short voice sample to analyze your speech pattern for early signs of cognitive decline.")

# Upload audio file
uploaded_file = st.file_uploader("Upload a voice file (.wav)", type=["wav"])

# New logic to analyze voice file
def analyze_audio(file_path):
    y, sr = librosa.load(file_path)

    if len(y) < 1:
        return 0

    duration = librosa.get_duration(y=y, sr=sr)

    if duration == 0:
        return 0

    # Energy calculation
    energy = librosa.feature.rms(y=y)[0]

    # Use median energy as baseline
    median_energy = np.median(energy)
    if median_energy == 0:
        median_energy = 1e-6

    # Pause detection
    pause_threshold = 0.2 * median_energy
    pauses = sum(energy < pause_threshold)
    pause_ratio = pauses / len(energy)

    # Speaking tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Scoring logic
    score = 100
    score -= pause_ratio * 40  # Penalty for frequent pauses
    score -= abs(tempo - 100) * 0.3  # Penalty for abnormal speaking rate

    # Keep score in 0–100 range
    score = int(max(0, min(score, 100)))
    return score

# Main logic: process and display results
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format="audio/wav")
    score = analyze_audio(tmp_path)

    # Display score
    st.subheader(f"🧪 Cognitive Health Score: **{score}/100**")

    # Display result message
    if score >= 75:
        st.success("🟢 Your speech pattern looks healthy.")
    elif score >= 50:
        st.warning("🟠 Some signs of cognitive slowing. Consider checking again later.")
    else:
        st.error("🔴 Speech shows early signs of decline. Medical follow-up is advised.")

    # Clean up temp file
    os.remove(tmp_path)
else:
    st.info("Please upload a short speech sample in WAV format.")
