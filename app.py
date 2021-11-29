import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from artia.NST_model import img_to_tensor, tensor_to_image
'''
# Neural Style Transfer front
'''

col1, col2 = st.columns(2)

content_uploaded_file = col1.file_uploader("Choose an content image:",
                                           type=["png", "jpeg"])

if content_uploaded_file is not None:
    content_img = Image.open(content_uploaded_file)

    col1.image(
        content_img,
        caption=f"You amazing image has shape",
        use_column_width=True,
    )

    content_img = np.array(content_img)
    content_img = img_to_tensor(content_img)

style_uploaded_file = col2.file_uploader("Choose a style image:",
                                         type=["png", "jpeg"])

if style_uploaded_file is not None:
    style_img = Image.open(style_uploaded_file)

    col2.image(
        style_img,
        caption=f"You amazing image has shape",
        use_column_width=True,
    )

    style_img = np.array(style_img)
    style_img = img_to_tensor(style_img)

st.image(tensor_to_image(content_img, style_img))
