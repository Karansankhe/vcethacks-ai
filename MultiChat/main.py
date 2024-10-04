from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
import requests
import base64
import tempfile

# Load environment variables from .env file
load_dotenv()

# Configure Google API with the provided API key
api_key = os.getenv("GENAI_API_KEY")
genai.configure(api_key=api_key)

# Sarvam API configuration
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_TTS_URL = os.getenv("SARVAM_TTS_URL")

# Initialize the Google Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to get a response from Google Gemini
def get_gemini_response(user_question):
    prompt = f"""
    You are a financial advisor chatbot. Answer the user's questions related to financial advice, investment strategies, and planning. 
    Provide clear and helpful information.

    User's Question: {user_question}
    """
    
    try:
        response = model.generate_content([user_question, prompt])
        return response.text
    except Exception as e:
        st.error(f"Failed to generate response: {str(e)}")
        return "I'm sorry, I couldn't process your request."

# Function for Sarvam Text-to-Speech conversion
def convert_text_to_speech(text, target_language="en-IN", speaker="meera", pitch=0, pace=1.65, loudness=1.5):
    # Truncate text to 500 characters if it exceeds the limit
    if len(text) > 500:
        st.warning("Input text exceeds 500 characters. It will be truncated.")
        text = text[:500]

    payload = {
        "inputs": [text],
        "target_language_code": target_language,
        "speaker": speaker,
        "pitch": pitch,
        "pace": pace,
        "loudness": loudness,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }
    headers = {'API-Subscription-Key': SARVAM_API_KEY}
    try:
        response = requests.post(SARVAM_TTS_URL, json=payload, headers=headers)

        if response.status_code == 200:
            json_data = response.json()
            if "audios" in json_data and json_data["audios"]:
                base64_string = json_data["audios"][0]
                wav_data = base64.b64decode(base64_string)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
                    wav_file.write(wav_data)
                    temp_filename = wav_file.name
                return temp_filename
            else:
                st.error("No audio data in the response.")
                st.json(json_data)  # Log the JSON response for debugging
        else:
            st.error(f"Failed to generate audio, status code: {response.status_code}, response: {response.text}")
            st.json(response.json())  # Log the JSON error response for debugging
    except Exception as e:
        st.error(f"Error during audio generation: {str(e)}")
    
    return None

# Initialize Streamlit app
st.set_page_config(page_title="Financial Q&A Bot", layout="wide")
st.title("Financial Q&A Bot")

# Chat input for user's question
user_question = st.text_input("Enter your question about financial planning or investments")

if st.button("Get Response"):
    if user_question:
        response = get_gemini_response(user_question)
        audio_file = convert_text_to_speech(response)  # Convert response to audio
        
        st.subheader("Bot Response")
        st.write(response)  # Display text response
        
        if audio_file:
            st.audio(audio_file)  # Play the audio response
        else:
            st.error("Audio generation failed.")
    else:
        st.error("Please enter a question.")
