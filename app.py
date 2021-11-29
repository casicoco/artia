
import streamlit as st

import streamlit as st
from PIL import Image


'''
# Neural Style Transfer front

This is a test for Neural Style Transfer

'''



st.title("Upload + Classification Example")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
image = Image.open(uploaded_file)
st.image(image, caption='Uploaded Image.', use_column_width=True)
st.write("")
st.write("Classifying...")
