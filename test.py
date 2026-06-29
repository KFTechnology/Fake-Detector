import streamlit as st 
import librosa
import torch
import numpy as np
from transformers import AutoModelForAudioClassification
from transformers import AutoFeatureExtractor
import os 
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speachsdk
from audiorecorder import audiorecorder
import tempfile


def load_model ():
    model = "garystafford/wav2vec2-deepfake-voice-detector"
    model_classifcation = AutoModelForAudioClassification.from_pretrained(model)
    model_extractor = AutoFeatureExtractor.from_pretrained(model)

    return model_classifcation, model_extractor

model_classifcation, model_extractor = load_model()


st.title("fake Audio detector")
st.write("please uplaod your Audio to know if fake or real ")

upload_file = st.file_uploader("Upload File" , type=["wav", "mp3", "flac"])

if upload_file is not None: 
   st.audio(upload_file)
   audio, sr = librosa.load(upload_file, sr=1600, mono=True)

max_lan = 1600 * 20 
audio = audio[:max_lan]

inputs = model_extractor(audio, sample_rate= 1600, return_tensors="pt", padding=True)

with torch.no_grad():
 outputs = model_classifcation(**inputs)
 
 probs = torch.nn.functional.softmax(outputs.logits, dim=-1)


prob_real = probs[0][0].item()
prob_fake = probs[0][1].item()

st.subheader("Result")

if prob_fake > prob_real:
        st.error("⚠️ FAKE AUDIO DETECTED")
else:
        st.success("✅ REAL HUMAN VOICE")

 
st.write(f"Real probability: {prob_real:.2%}")
st.write(f"Fake probability: {prob_fake:.2%}")


st.progress(float(prob_fake))
