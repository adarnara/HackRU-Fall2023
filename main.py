import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Skin.AI")
st.write("Vaishnav Venkat, Vignesh Venkat, Pranav Patil, Adarsh Narayan")

label = "Upload an image of your skin or use camera to take a photo:"
st.file_uploader(label)

img_file_buffer = st.camera_input("Take a picture")
if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    st.write(type(cv2_img))

   # st.write(cv2_img.shape)

