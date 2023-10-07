import streamlit as st
import cv2
import numpy as np
from PIL import Image

@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    return img
st.set_page_config(page_title="Skin.AI")
st.title("Skin.AI")
st.write("*Vaishnav Venkat, Vignesh Venkat, Pranav Patil, Adarsh Narayan*")

photo_upload = st.radio("Would you like to upload an image or take a photo now:", ["Upload an image", "Take a photo now"])
if photo_upload == "Upload an image":
    label = "Upload an image of your skin or use camera to take a photo(only .png and .jpeg files are accepted):"
    file = st.file_uploader(label, type=["png", "jpeg"])
    if file is not None:
        st.image(load_image(file))
else:
    file = st.camera_input("Take a picture")
    if file:
        st.image(file)

if file is not None:
    bytes_data = file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img_bytes = (type(cv2_img))

    #st.write(cv2_img)

st.header("Your Diagnosis:")
#st.write()

