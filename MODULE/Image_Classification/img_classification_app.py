import pickle
from PIL import Image
from io import BytesIO
from img2vec_pytorch import Img2Vec
import streamlit as st
import os

#NOTE don't forget to upload the picke (model) file to your Google Colab First
#to run this code
#you can use any model that is capable of classifiying images that uses img2vec_pytorch
current_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(current_dir, 'model_needs_npk.p')
with open(filename, 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

## Streamlit Web App Interface
st.set_page_config(layout="wide", page_title="Cat Breed's Image Classification")

st.write("## This is a demo of an Image Classification Model in Python!")
st.write(
    ":grin: We'll try to predict the image on what features it was trained via the uploaded image :grin:"
)
st.sidebar.write("## Upload and download :gear:")

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
    img = Image.open(my_upload)
    features = img2vec.get_vec(img)
    pred = model.predict([features])

    # print(pred)
    col2.header(pred)
    # st.sidebar.markdown("\n")
    # st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "jfif"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
else:
    st.write("by koalatech...")