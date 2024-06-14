
import streamlit as st
# Provide the code block for reference
st.header('Image Classification App')
st.subheader('This python code is implemented for Streamlit')
st.code('''
import pickle
from PIL import Image
from io import BytesIO
from img2vec_pytorch import Img2Vec
import streamlit as st

# Load the pre-trained model
with open('./MODULE/Image_Classification/model_needs_npk.p', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

# Streamlit Web App Interface
st.set_page_config(layout="wide", page_title="Cat Breed's Image Classification")

st.write("## This is a demo of an Image Classification Model in Python!")
st.write(":grin: We'll try to predict the image on what features it was trained via the uploaded image :grin:")
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Function to convert image to bytes
@st.cache_data
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return byte_im

# Function to process and classify the image
def fix_image(upload):
    image = Image.open(upload)
    col1.write("Image to be predicted :camera:")
    col1.image(image)

    col2.write("Category :wrench:")
    img = Image.open(upload)
    features = img2vec.get_vec(img)
    pred = model.predict([features])

    col2.header(pred[0])

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "jfif"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
else:
    st.write("by koalatech...")
''')
