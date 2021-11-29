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

    test_1 = col1.image(content_img,
                        caption='test',
                        channels="RGB",
                        output_format='PNG',
                        use_column_width=True)

    img = content_img
    img.resize((299, 299), Image.ANTIALIAS)
    content_img = np.array(img)[:, :, 0:3].astype(float) / 255

style_uploaded_file = col2.file_uploader("Choose an style image:",
                                         type=["png", "jpeg"])

if style_uploaded_file is not None:
    style_img = Image.open(style_uploaded_file)

    test_1 = col2.image(style_img,
                        caption='test',
                        channels="RGB",
                        output_format='PNG',
                        use_column_width=True)

    img = style_img
    img.resize((299, 299), Image.ANTIALIAS)
    style_img = np.array(img)[:, :, 0:3].astype(float) / 255.

    st.image(tensor_to_image(content_img, style_img))
