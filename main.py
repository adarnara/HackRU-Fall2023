import streamlit as st
import cv2
import numpy as np

st.title("Skin.AI")
st.write("Vaishnav Venkat, Vignesh Venkat, Pranav Patil, Adarsh Narayan")

label = "Upload an image of your skin or use camera to take a photo:"
st.file_uploader(label)

picture = st.camera_input("Use camera:")

if picture:
    st.image(picture)


