
import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title, hide_pages

# Setting the title and pages layout
add_page_title()
show_pages(
    [
        Page("./home.py", "ITEQMT Machine Learning Application Portfolio", "💻"),

        Section("Machine Learning UI App", "🧙‍♂️"),
        Page("./MODULE/Prediction/taylor_discography_app.py", "Taylor Swift Discography", "1️⃣", in_section=True),
        Page("./MODULE/Sentiment_Analysis/emotion_classifier_model_app.py", "Emotion Analyzer", "2️⃣", in_section=True),
        Page("./MODULE/Image_Classification/img_classification_app.py", "Cats Breed Identifier", "3️⃣", in_section=True),

        Section("Sample Source Code", "💾"),
        Page("./MODULE/Prediction/taylor_discography_src.py", "Taylor Swift Discography SRC", "1️⃣", in_section=True),
        Page("./MODULE/Sentiment_Analysis/emotion_classifier_model_scr.py", "Emotion Analyzer SRC", "2️⃣", in_section=True),
        Page("./MODULE/Image_Classification/img_classification_src.py", "Cats Breed Identifier SRC", "3️⃣", in_section=True),
    ]
)

hide_pages(["Thank you"])

# Main Portfolio Page Content
st.markdown("## ITEQMT Machine Learning Application Portfolio")
st.markdown("### ML Learning by Marianne Reyne Madrona")
st.image("./back.jpg")

st.markdown("<hr>", unsafe_allow_html=True)

# About Myself Section
st.markdown("### About Myself")
st.write("""
I am Marianne Reyne Madrona, currently a third-year student pursuing a Bachelor of Science in Information Systems. I have a passion for learning and enjoy gaining knowledge from others.

I have a deep appreciation for music, particularly Taylor Swift's songs, which I enjoy listening to regularly. Additionally, I have a fondness for cats and find joy in their company.
""")
st.markdown("#### Skills")
st.write("""
* Programming: Beginner level in PHP; familiar with Python.
* Communication Skills: Proficient in both oral and written English and Filipino.
* Software Proficiency: Experienced with Microsoft Office Suite (Word, Excel, PowerPoint).
* Creative Abilities: Capable of designing, drawing, and painting.
""")

st.markdown("<hr>", unsafe_allow_html=True)

# Description of Applications Section
st.markdown("### Description of Applications")

# Taylor Swift Discography
st.markdown("#### Taylor Swift Song Predictor")
st.write("""
The Taylor Swift Song Predictor is a Streamlit-based web app where users input song characteristics like loudness, speechiness, and acousticness to find the most similar Taylor Swift song. It utilizes a pre-trained machine learning model to make predictions, providing a user-friendly way to explore similarities in Taylor Swift's discography based on specific musical attributes.
""")

# Emotion Analyzer
st.markdown("#### Emotion Analyzer")
st.write("""
This Streamlit-based app, "Emotion Analyzer," predicts emotions based on user-inputted messages. Users type their feelings, and the app uses a pre-trained Naive Bayes classifier to detect emotions like Happy 😊, Sad 😢, Angry 😡, Excited 🤩, Nervous 😰, or Scared 😱. It provides instant feedback with an emoji that corresponds to the detected emotion, making it a quick and interactive tool for understanding and expressing emotions.
""")

# Cats Breed Identifier
st.markdown("#### Cats Breed Identifier")
st.write("""
This app classifies uploaded cat images into three breeds: Persian, Sphynx, and Tonkinese. It uses a trained model to analyze visual features and predict the most likely breed, offering a straightforward tool for identifying different cat breeds based on uploaded photos.
""")

st.info("Please note that certain functionalities may be limited due to hosting constraints on Streamlit.")

st.markdown("### What I Learned")
st.write("""
In class, I've gained hands-on experience with Python scripting, where I've learned to prepare datasets, profile data, and classify images. I've enjoyed building datasets by collecting numerous cat images and mastering the training process. Concepts like pickling and CSV for data storage and retrieval have become clear to me, enriching my understanding of practical data handling techniques. I've also learned the importance of patience, especially when working with large datasets and training models.
""")

st.markdown("<hr>", unsafe_allow_html=True)

# Footer Information
st.markdown("### 👨‍🔧 ML Learning by [Your Name](https://github.com/your-github-username)")

st.info("Visit the project [Github](https://github.com/your-github-username/streamlit_web_app)")

st.info("👨‍🔧 Please note that certain functionalities may be limited due to hosting constraints on Streamlit.")

# Custom CSS
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
h1, h2, h3, h4, h5, h6 {
    color: #4A90E2;
}
hr {
    border: 1px solid #4A90E2;
}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
