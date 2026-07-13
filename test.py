import streamlit as st
import azure.cognitiveservices.speech as speechsdk
import os

# ===== CONFIG =====
speech_key = "YOUR_KEY_HERE"
region = "eastus2"

speech_config = speechsdk.SpeechConfig(
    subscription=speech_key,
    region=region
)

# ===== VOICE OPTIONS =====
voices = {
    "Friendly Female (Jenny)": "en-US-JennyNeural",
    "Professional Female (Aria)": "en-US-AriaNeural",
    "Male Voice (Guy)": "en-US-GuyNeural",
    "Arabic Female (Zariyah)": "ar-SA-ZariyahNeural"
}

# ===== UI =====
st.title("🎙️ AI Voice Generator")

text = st.text_area("Enter text", "Hello Farah, this is your AI voice!")

voice_choice = st.selectbox("Choose Voice", list(voices.keys()))
style = st.selectbox("Emotion", ["general", "cheerful", "sad", "angry"])

rate = st.slider("Speed", 0.5, 1.5, 1.0)
pitch = st.slider("Pitch", -20, 20, 0)

# Convert to SSML format values
rate_str = f"{rate}"
pitch_str = f"{pitch}%"

selected_voice = voices[voice_choice]

# ===== GENERATE BUTTON =====
if st.button("Generate Voice 🔊"):

    ssml = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
           xmlns:mstts="http://www.w3.org/2001/mstts"
           xml:lang="en-US">
        <voice name="{selected_voice}">
            <mstts:express-as style="{style}">
                <prosody rate="{rate_str}" pitch="{pitch_str}">
                    {text}
                </prosody>
            </mstts:express-as>
        </voice>
    </speak>
    """

    audio_file = "output.wav"

    audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_file)

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )
    


    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        st.success("✅ Audio generated!")
        st.audio(audio_file)
    else:
        st.error("❌ Error generating speech")