import streamlit as st
import tempfile
import os
import google.generativeai as genai

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

user_mesage = ("Following the transcript from a recorded call \n"
"<TRANSCRIPTION> \n" 
"   {transcript} \n" 
"</TRANSCRIPTION> \n" 
"#OBJECTIVE \n" 
"Please analyse the call and answer following questions in bullet points. \n" 
"Question 1 - Please provide detail summary of the call \n" 
"Question 2 - If there are any coverages being discussed on the call. \n" 
"Question 3 - If there is any kind of complaint or unhappiness being expressed in the call \n" 
"Question 4 - If there seems to be any kind of confusion or misunderstanding in the conversation \n")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

openai_model = ChatOpenAI(model="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a smart audio transcript analyst. Your task is to analyse the transcript of a call conversation between an insurance company support engineer and insurance customer. "),
    ("user", user_mesage)
])

output_parser = StrOutputParser()

questions = [
    "Please summarize the conversation"
]

# Configure Google API for audio summarization
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

def summarize_audio(audio_file_path):
    """Summarize the audio using Generative AI."""

    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    audio_file = genai.upload_file(path=audio_file_path)

    response = model.generate_content(
        [
            "Please transcribe the following audio in English.",
            audio_file
        ]
    )
    transcript = response.text

    chain = prompt | openai_model | output_parser

    output = chain.invoke({"transcript": transcript})

    return output

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

# Streamlit app interface
st.title('Support Call Analysis')

audio_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])
if audio_file is not None:
    audio_path = save_uploaded_file(audio_file)  # Save the uploaded file and get the path
    st.audio(audio_path)

    if st.button('Summarize Audio'):
        with st.spinner('Summarizing...'):
            answer = summarize_audio(audio_path)
            st.subheader('Detailed Call Analsysis')
            st.info(answer)
