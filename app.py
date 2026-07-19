import streamlit as st
import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from audiorecorder import audiorecorder
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import tempfile
import os
import shutil

load_dotenv()

st.write("FFmpeg:", shutil.which("ffmpeg"))

# Speech-text-speech model 
Azure_Model_Endpoint =  os.getenv("AZURE_Resource_ENDPOINT")
Azure_API_Key = os.getenv("AZURE_SPEECH_KEY")
Azure_Region = os.getenv("AZURE_SPEECH_REGION")
Azure_Model_Name = os.getenv("AZURE_MOdel_Name")


config_speech = speechsdk.SpeechConfig(
   subscription=Azure_API_Key, 
   region=Azure_Region
)

Voices = {
    "English Voice (US)" : {
    "Friendly Female": "en-US-JennyNeural",
    "Professional Female": "en-US-AriaNeural",
    "Warm Female": "en-US-AmberNeural",
    "Casual Female": "en-US-MichelleNeural",
    "Friendly Male": "en-US-GuyNeural",
    "Deep Male": "en-US-DavisNeural",
    "Calm Male": "en-US-JasonNeural",
    "Young Male": "en-US-TonyNeural"
    }, 

    "Arabic Voice" : {
     "Arabic Female (Saudi)": "ar-SA-ZariyahNeural",
    "Arabic Male (Saudi)": "ar-SA-HamedNeural",
    "Arabic Female (UAE)": "ar-AE-FatimaNeural",
    "Arabic Male (UAE)": "ar-AE-HamdanNeural",
    "Arabic Female (Egypt)": "ar-EG-SalmaNeural",
    "Arabic Male (Egypt)": "ar-EG-ShakirNeural" 
    },

    "English Voice (UK)" : {
    "UK Female (Soft)": "en-GB-LibbyNeural",
    "UK Female (Bright)": "en-GB-SoniaNeural",
    "UK Male": "en-GB-RyanNeural" 
    },
    "Spanish Voice" : {
    "Spanish Female": "es-ES-ElviraNeural",
    "Spanish Male": "es-ES-AlvaroNeural"
    }, 
     "French Vice" : {
    "French Female": "fr-FR-DeniseNeural",
    "French Male": "fr-FR-HenriNeural"
     }, 
     "German Voice" : {
    "German Female": "de-DE-KatjaNeural",
    "German Male": "de-DE-ConradNeural"
     }, 
     "Multi Language Use voice" : {
    "Expressive Female": "en-US-JennyMultilingualNeural",
    "Expressive Male": "en-US-GuyNeural",
    "Narration Style": "en-US-AriaNeural"
     }
}


# load model from Hugging Face by creating model name, model and extractor 
@st.cache_resource
def load_model():
    model_name = "garystafford/wav2vec2-deepfake-voice-detector"
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
        # return model and extractor in method load model to return the model name 
    return model, extractor

model, extractor = load_model()


# create frontend by using streamlit website 
st.title("🎤 AI Fake Audio Detector")
st.write("Upload OR record audio to check if it is REAL or FAKE")

option = st.radio("Choose input method:", ["Generate Text-Speech Audio","Upload Audio", "Record Audio"], horizontal=True, width="stretch")

audio_data = None

if option == "Generate Text-Speech Audio":
 
 Text_Area = st.text_area("Enter your text here", height=250)

 Language_choice = st.selectbox("Choose Language", Voices.keys())
 voices_choice = st.selectbox("Choose Voice",list(Voices[Language_choice].keys())) 
 Selected_Voice = Voices[Language_choice][voices_choice]
 config_speech.speech_synthesis_voice_name = Selected_Voice

 Generate_Button = st.button("Generate Voice 🔊")

 if (Generate_Button): 
      ssml = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
           xmlns:mstts="http://www.w3.org/2001/mstts"
           xml:lang="en-US">
        <voice name="{Selected_Voice}">
            <mstts:express-as style ="friendly">
                <prosody rate="0%" pitch="0%">
                    {Text_Area}
                </prosody>
            </mstts:express-as>     
        </voice>
    </speak>
    """
      
      audio_file = "Output.wav"
      config_audio = speechsdk.audio.AudioOutputConfig(filename=audio_file)
      synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=config_speech,
        audio_config=config_audio
        )
      
      result = synthesizer.speak_ssml_async(ssml).get()

      if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted: 
          st.success("✅ Audio generated!")
          st.audio(audio_file)
          audio_data = audio_file

      elif result.reason == speechsdk.ResultReason.Canceled:
         details = result.cancellation_details
         st.error(f"❌ Reason: {details.reason}")
         st.error(f"Error details: {details.error_details}")

      else:
          st.warning(f"⚠️ Unexpected result: {result.reason}")
           
          
 
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


if audio_data is not None:
    try:
       
        audio, sr = librosa.load(audio_data, sr=16000, mono=True)    
        audio = audio[:16000 * 15]
 
        inputs = extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

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




