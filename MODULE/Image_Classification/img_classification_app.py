import pickle
from img2vec_pytorch import Img2Vec
from PIL import Image
import streamlit as st
from io import BytesIO
import os

# File path for the pickled model
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, 'model_needs_npk.p')

# Load the model
try:
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Error: File '{file_path}' not found.")
except Exception as e:
    st.error(f"Error opening file '{file_path}': {str(e)}")

img2vec = Img2Vec()

# Streamlit Web App Interface
st.set_page_config(layout="wide", page_title="Cat Breeds Classifier")

st.write("## This is a demo of an Image Classification Model in Python!")
st.write(":grin: We'll try to predict the image on what features it was trained via the uploaded image :grin:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Function to convert image to bytes
@st.cache(allow_output_mutation=True)
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return byte_im

# Function to process uploaded image
def fix_image(upload):
    image = Image.open(upload)
    col1.write("Image to be predicted :camera:")
    col1.image(image)

    col2.write("Category :wrench:")
    img = Image.open(upload)
    features = img2vec.get_vec(img)
    pred = model.predict([features])

    col2.header(pred)

col1, col2 = st.columns(2)
my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "jfif"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
else:
    st.write("by reyne...")
