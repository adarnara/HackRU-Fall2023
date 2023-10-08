import streamlit as st
import os



st.set_page_config(layout = "wide")
current_directory = os.getcwd()
st.write(current_directory)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Vignesh Venkat")
    image_path = os.path.join(current_directory, "Pages/viggy.jpg")
    st.image(image_path)

with col2:
    st.subheader("Pranav Patil")
    image_path = os.path.join(current_directory, "Pages/pacho.jpeg")
    st.image(image_path)

with col3:
    st.subheader("Adarsh Narayanan")
    image_path = os.path.join(current_directory, "Pages/addy.jpeg")
    st.image(image_path)



with col4:
    st.subheader("Vaishnav Venkat")
    image_path = os.path.join(current_directory, "Pages/vaichuuu.jpeg")
    st.image(image_path)
