import streamlit as st
import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from audiorecorder import audiorecorder
import tempfile


# load model from Hugging Face by creating model name, model and extractor 
st.cache_resource
def load_model():
    model_name = "garystafford/wav2vec2-deepfake-voice-detector"
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    # return model and eztractor in method load model to return the model name 
    return model, extractor

model, extractor = load_model()

# create and use device whoch can work if the user donot have processor of cpu or gpu 
device = "cuda" 
if torch.cuda.is_available (): 
   torch.device.type
else: 
    torch.device  = "cuda"   
      
inputs = extractor(audio, return_tensor="pt").to(device) # type: ignore


# create frontend by using streamlit website 
st.title("🎤 AI Fake Audio Detector")
st.write("Upload OR record audio to check if it is REAL or FAKE")

option = st.radio("Choose input method:", ["Upload Audio", "Record Audio", "Record Vedio"], horizontal=True, width="stretch")

audio_data = None

if option == "Upload Audio":
    uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3", "flac"])
    
    if uploaded_file:
        st.audio(uploaded_file)
        audio_data = uploaded_file

         

if option == "Record Audio":
    audio = audiorecorder("Start Recording", "Stop Recording")
    
    if len(audio) > 0:
        st.audio(audio.export().read())
        

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio.export().read())
            audio_data = f.name


if option == "Record Vedio": 
  vedio = st.camera_input
  if len(vedio) > 0: 
     st.vedio.read() 
     st.vedio(vedio.__defaults__)

if audio_data is not None:
    try:
       
        audio, sr = librosa.load(audio_data, sr=16000, mono=True)
      
        audio = audio[:16000 * 15]

        sr = sr[device]
        
        inputs = extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

        adio1, sr1 = librosa.load(audio, sr=16000, n_mfcc=13)

        
      
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        prob_real = probs[0][0].item()
        prob_fake = probs[0][1].item()

        st.subheader("Result")

        if prob_fake > prob_real:
            st.error(f"⚠️ FAKE AUDIO ({prob_fake:.2%})")
        else:
            st.success(f"✅ REAL VOICE ({prob_real:.2%})")

        st.write(f"Real: {prob_real:.2%}")
        st.write(f"Fake: {prob_fake:.2%}")

        st.progress(float(prob_fake))

    except Exception as e:
        st.error(f"Error processing audio: {e}")



    

     



