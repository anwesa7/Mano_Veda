import streamlit as st
import librosa
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(page_title="ManoVeda: Alzheimer’s Voice Screening", layout="centered")

# ----------------------------
# 💄 Custom CSS for Enhanced UI Styling
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #eef2f7;
    font-family: 'Segoe UI', sans-serif;
}

h1, h2, h3, h4 {
    font-family: 'Segoe UI', sans-serif;
    color: #2d3436;
}

.stApp {
    max-width: 860px;
    margin: auto;
    padding-top: 2rem;
}

.container-box {
    background: linear-gradient(to top left, #ffffff, #f7f9fc);
    border-radius: 16px;
    padding: 32px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.07);
    margin-bottom: 2rem;
}

.report-box {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #e0e0e0;
    margin-top: 1rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
}

.disclaimer {
    font-size: 0.85rem;
    color: #888888;
    text-align: center;
    padding: 2rem 0;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 🧠 Header
# ----------------------------
st.markdown('<div class="container-box">', unsafe_allow_html=True)
st.title("🧠 ManoVeda: Alzheimer’s Voice Screening")
st.markdown("""
Welcome to the **early detection assistant**. This AI-powered app evaluates your speech pattern for early signs of cognitive changes.
Simply upload a `.wav` voice file to get started.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# 📤 Upload Section
# ----------------------------
st.markdown('<div class="container-box">', unsafe_allow_html=True)
st.subheader("🎙 Upload Your Voice Sample")
uploaded_file = st.file_uploader("Upload a short .wav audio sample (10–20 sec)", type=["wav"])
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# 🔍 Analyze Function
# ----------------------------
def analyze_audio(file_path):
    try:
        y, sr = librosa.load(file_path)
        if len(y) < 1:
            return 0, 0, 0, np.array([])

        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 1:
            return 0, duration, 0, np.array([])

        energy = librosa.feature.rms(y=y)[0]
        median_energy = np.median(energy)
        median_energy = median_energy if median_energy > 0 else 1e-6

        pause_threshold = 0.2 * median_energy
        pauses = sum(energy < pause_threshold)
        pause_ratio = pauses / len(energy)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

        score = 100
        score -= pause_ratio * 40
        score -= abs(tempo - 100) * 0.3
        score = int(max(0, min(score, 100)))

        return score, duration, tempo, energy
    except Exception as e:
        st.error(f"⚠ Error: {e}")
        return 0, 0, 0, np.array([])

# ----------------------------
# 🧪 Result Section
# ----------------------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.markdown('<div class="container-box">', unsafe_allow_html=True)
    st.audio(uploaded_file, format="audio/wav")

    score, duration, tempo, energy = analyze_audio(tmp_path)

    st.subheader("📊 Voice Analysis Report")
    st.markdown(f"""
    <div class="report-box">
        <strong>Cognitive Health Score:</strong> <span style="color:#2980b9; font-size: 1.6rem">{score}/100</span><br><br>
        <ul>
            <li><strong>Audio Duration:</strong> {duration:.2f} sec</li>
            <li><strong>Speaking Tempo:</strong> {tempo:.2f} BPM</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if score >= 75:
        st.success("🟢 Your voice shows no signs of cognitive decline.")
    elif score >= 50:
        st.warning("🟠 Mild signs of slowing. Try again after a week or seek professional screening.")
    else:
        st.error("🔴 Signs of early decline detected. Consider visiting a specialist.")

    if energy.size > 0:
        st.markdown("### 📈 Voice Energy Pattern")
        fig, ax = plt.subplots()
        ax.plot(energy, color='#6a5acd', linewidth=1.5, label="RMS Energy")
        ax.axhline(y=0.2 * np.median(energy), color='red', linestyle='--', label='Pause Threshold')
        ax.set_title("Short-Time Energy")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Energy Level")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("No usable energy data found in the uploaded audio.")

    st.markdown('</div>', unsafe_allow_html=True)
    os.remove(tmp_path)
else:
    st.markdown('<div class="container-box">', unsafe_allow_html=True)
    st.info("👆 Please upload a short .wav voice sample to begin.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# 📌 Disclaimer
# ----------------------------
st.markdown("""
<div class="disclaimer">
⚠ This tool is intended for **early screening** only. It is not a replacement for medical evaluation. Please consult a neurologist for a diagnosis.
</div>
""", unsafe_allow_html=True)
