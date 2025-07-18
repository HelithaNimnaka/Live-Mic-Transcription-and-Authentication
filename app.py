import streamlit as st
import sounddevice as sd
import soundfile as sf
import tempfile
import numpy as np
import os

from functions import manual_authentication, transcribe
from Speaker_Authontication import verify_speakers, extract_embedding
from voice_enhancement import voice_enhancement

# ── CONFIG ────────────────────────────────────────────────────────────
USER_CSV        = "speakers/users.csv"
SPEAKER_DB      = "speakers"
EXPECTED_PHRASE = "Hello I need to activate my voice"
EXPECTED_PHRASE2 = "Hello I need to activate my voice."
N_REPEATS       = 5
SR              = 16000
# ───────────────────────────────────────────────────────────────────────

st.title("🎤 Live Mic Transcription + Authentication")

# Initialize session state
st.session_state.setdefault("wav_path", None)
st.session_state.setdefault("enrol_step", 0)
st.session_state.setdefault("enrol_embeds", [])

# Global record duration slider
DURATION = st.slider("🎙️ Record duration (seconds)", 1, 10, 3)

# ╭──────────────── Voice-authentication quick test ─────────────────╮
with st.expander("🔊 Voice Authentication (one-shot)"):
    st.subheader("VoicePrint")
    st.write("Press the button to record once and test recognition.")

    if st.button("🎙️ Record once"):
        st.info("Recording…")
        rec = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
        sd.wait()

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, rec, SR)

        # Enhance the audio before storing and playing
        enhanced_path = voice_enhancement(tmp.name)
        st.session_state.wav_path = enhanced_path
        st.audio(enhanced_path)  # Now plays enhanced audio
        st.success("Recording complete.")

# ── Transcribe + identify that one sample ───────────────────────────
if st.session_state.wav_path:
    st.header("🧠 Processing…")

    enhanced_path = st.session_state.wav_path  # Use already enhanced audio

    with st.spinner("Transcribing…"):
        txt = transcribe(enhanced_path)
        st.write("**Transcription:**", txt)

        with open("transcripts.txt", "a") as f:
            f.write(txt + "\n")

    st.success("Transcript saved.")

    with st.spinner("Authenticating speaker…"):
        res = verify_speakers(enhanced_path, speaker_db=SPEAKER_DB)
        if res["status"] == "VERIFIED":
            st.success(f"🔐 Speaker: {res['match']} (similarity {res['similarity']:.2f})")
        else:
            st.warning("⚠️ Speaker not recognized.")

# ╭──────────── 5-shot Voice-print enrolment (new UI) ────────────────╮
st.subheader("📋 Enrol a new speaker (5× same phrase)")

step = st.session_state.enrol_step
embs = st.session_state.enrol_embeds

st.progress(len(embs) / N_REPEATS, text=f"{len(embs)}/{N_REPEATS} successful recordings captured")

if step < N_REPEATS and st.button("🎙️ Record a phrase"):
    st.info(f"Say exactly: **{EXPECTED_PHRASE}**")

    rec = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
    sd.wait()

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, rec, SR)

    # Enhance
    enhanced_path = voice_enhancement(tmp.name)
    st.audio(enhanced_path)

    # Transcribe
    spoken = transcribe(enhanced_path).lower().strip()
    st.write("**Transcription:**", spoken)
    st.session_state.wav_path = enhanced_path

    if EXPECTED_PHRASE not in spoken and EXPECTED_PHRASE2 not in spoken:
        st.error("❌ Phrase didn’t match – try again.")
    else:
        # Extract embedding for this recording
        emb = extract_embedding(enhanced_path)
        embs.append(emb)
        st.session_state.enrol_step += 1
        st.success("✅ Take accepted.")

# After 5 recordings
if step == N_REPEATS:
    avg_embed = np.mean(np.stack(embs), axis=0)
    new_name = st.text_input("Label for this voice-print:")

    if st.button("💾 Save voice-print") and new_name:
        # Save the averaged embedding
        os.makedirs(SPEAKER_DB, exist_ok=True)
        final_path = os.path.join(SPEAKER_DB, f"{new_name}.npy")
        np.save(final_path, avg_embed)
        
        st.success(f"Enrolled new speaker: {new_name}")
        st.session_state.enrol_step = 0
        st.session_state.enrol_embeds = []

# ╭───────────────────── Manual login / enrolment ────────────────────╮
with st.expander("🔐 Manual Login / Enrollment"):
    st.subheader("Login")
    with st.form("manual_form"):
        u = st.text_input("Username:")
        p = st.text_input("Password:", type="password")
        if st.form_submit_button("Login"):
            if u and p:
                auth = manual_authentication(u, p, manual_csv=USER_CSV)
                st.success("✅ Authenticated.") if auth == "VERIFIED" else st.error("❌ Invalid credentials.")
            else:
                st.warning("Both fields required.")

    st.subheader("Enroll new password user")
    with st.form("enroll_form"):
        nu = st.text_input("Choose a username:")
        npw = st.text_input("Choose a password:", type="password")
        if st.form_submit_button("Add user"):
            if nu and npw:
                with open(USER_CSV, "a") as f:
                    f.write(f"\n{nu},{npw}")
                st.success(f"✅ Added user '{nu}'.")
            else:
                st.warning("Both fields required.")
