import streamlit as st

st.set_page_config(layout = "wide")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Vignesh Venkat")
    st.image("/Users/vaishnavvenkat/PycharmProjects/HackRU-Fall2023/Pages/viggy.jpg")

with col2:
    st.subheader("Pranav Patil")
    st.image("/Users/vaishnavvenkat/PycharmProjects/HackRU-Fall2023/Pages/pacho.jpeg")

with col3:
    st.subheader("Adarsh Narayanan")
    st.image("/Users/vaishnavvenkat/PycharmProjects/HackRU-Fall2023/Pages/addy.jpeg")



with col4:
    st.subheader("Vaishnav Venkat")
    st.image("/Users/vaishnavvenkat/PycharmProjects/HackRU-Fall2023/Pages/vaichuuu.jpeg")
