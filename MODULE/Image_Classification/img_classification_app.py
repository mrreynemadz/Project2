import pickle
from img2vec_pytorch import Img2Vec
from PIL import Image
import streamlit as st
from io import BytesIO
import os
import sklearn

# Set the page configuration for the Streamlit app
st.set_page_config(layout="wide", page_title="Cat Breeds Classifier")

# Write the title and description for the app
st.write("## This is a demo of an Image Classification Model in Python!")
st.write(
    ":grin: We'll try to predict the image on what features it was trained via the uploaded image :grin:"
)

# Print the current working directory for debugging
st.text(f"Current working directory: {os.getcwd()}")

# Custom Unpickler to handle _Scorer attribute error
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'sklearn.metrics._scorer' and name == '_Scorer':
            from sklearn.metrics import _scorer
            return _scorer._Scorer
        return super().find_class(module, name)

# Load the model from the specified path
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'model_needs_npk.p')
model = None

if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        try:
            model = CustomUnpickler(f).load()
            st.text("Model loaded successfully")
        except Exception as e:
            st.text(f"Error loading model: {e}")
else:
    st.text(f"File not found: {file_path}")

# Initialize Img2Vec for image feature extraction
img2vec = Img2Vec()

# Define the maximum file size for uploads (5MB)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Function to convert image to bytes for downloading
@st.cache_data
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="jpg")
    byte_im = buf.getvalue()
    return byte_im

# Function to process the image and make predictions
def fix_image(upload):
    if model is None:
        st.error("Model is not loaded. Please check the model file.")
        return

    image = Image.open(upload)
    col1.write("Image to be predicted :camera:")
    col1.image(image)

    col2.write("Category :wrench:")
    features = img2vec.get_vec(image)

    try:
        pred = model.predict([features])
        col2.header(pred)
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Create columns in the Streamlit app for layout
col1, col2 = st.columns(2)

# File uploader widget for uploading images
my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "jfif"])

# Process the uploaded file if it exists
if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
else:
    st.write("by reyne...")
