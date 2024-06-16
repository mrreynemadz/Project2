import pickle
from img2vec_pytorch import Img2Vec
from PIL import Image
import streamlit as st
# from rembg import remove
from PIL import Image
from io import BytesIO
import base64
import os

#NOTE don't forget to upload the picke (model) file to your Google Colab First
#to run this code
#you can use any model that is capable of classifiying images that uses img2vec_pytorch

file_path = 'model_needs_npk.p'

if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        try:
            model = pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file: {e}")
else:
    print(f"File not found: {file_path}")

img2vec = Img2Vec()

# Streamlit Web App Interface
st.set_page_config(layout="wide", page_title="Cat Breeds Classifier")

st.write("## This is a demo of an Image Classification Model in Python!")
st.write(
    ":grin: We'll try to predict the image on what features it was trained via the uploaded image :grin:"
)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Download the fixed image
@st.cache_data
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="jpg")
    byte_im = buf.getvalue()
    return byte_im

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
