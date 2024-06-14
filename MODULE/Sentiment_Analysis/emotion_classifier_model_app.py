import streamlit as st
import pandas as pd
import pickle
from nltk.corpus import names
import os

# Define the layout
st.markdown("""
    <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 15px; text-align: center;">
        <h2 style="color: #4CAF50;">Emotion Analyzer</h2>
        <label for="message" style="font-size: 16px; font-weight: bold;">Tell me what you feel today:</label>
    </div>
""", unsafe_allow_html=True)

# Input box for the user's message
message = st.text_input("")

# Load the trained Naive Bayes classifier from the saved file
current_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(current_dir, 'emotion_classifier_model.sav')
loaded_model = pickle.load(open(filename, 'rb'))

# Define features (words) and their corresponding labels (emotions)
@st.cache_data
def word_features(words):
    return dict([(word, True) for word in words])

# Define a function for your button click
def sayFeeling():
    if message:
        message_tone = loaded_model.classify(word_features(message.split())).capitalize()
        emoji = ""
        if message_tone == 'Happy':
            emoji = "😊"
        elif message_tone == 'Sad':
            emoji = "😢"
        elif message_tone == 'Angry':
            emoji = "😡"
        elif message_tone == 'Excited':
            emoji = "🤩"
        elif message_tone == 'Nervous':
            emoji = "😰"
        elif message_tone == 'Scared':
            emoji = "😱"

        st.markdown(f"""
            <div style="margin-top: 20px; text-align: center;">
                <h2 style="font-weight: bold;">Emotion detected: {message_tone}</h2>
                <h1 style='font-size: 100px;'>{emoji}</h1>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.write("Please enter a message to analyze.")

# Button for submitting the input
st.button('Say it', on_click=sayFeeling)

# To run on terminal issue this command:
# streamlit run emotion_classifier_model_app.py
