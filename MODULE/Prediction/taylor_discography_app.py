import streamlit as st
import pandas as pd
import pickle

# Load the trained model from the saved file
filename = './taylor_discography_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Define a function to use the model to make predictions
def predict_song(features):
    song_name = loaded_model.predict([features])[0]
    st.text(f"The predicted song is: {song_name}")

# Streamlit UI
st.title("Taylor Swift Song Predictor :musical_note:")
st.subheader("Enter song features to determine the most similar Taylor Swift song:")

# Get user input for song features
loudness_input = st.slider("Loudness: ", -60.0, 0.0)
speechiness_input = st.slider("Speechiness: ", 0.0, 1.0)
acousticness_input = st.slider("Acousticness: ", 0.0, 1.0)

# Prepare the input features for prediction
if st.button("Predict Song"):
    features = [loudness_input, speechiness_input, acousticness_input]
    predict_song(features)

st.text("The song suitable for these features will be displayed above.")
