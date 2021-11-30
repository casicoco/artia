import streamlit as st
import matplotlib.pyplot as plt
from artia.NST_model import img_to_tensor, tensor_to_image
import requests
from PIL import Image
import io
import numpy as np

'''
# Neural Style Transfer front
'''

col1, col2 = st.columns(2)
st.set_option("deprecation.showfileUploaderEncoding", False)
content_uploaded_file = col1.file_uploader("Choose an content image:",
                                           type=["png", "jpeg",'jpg'])

if content_uploaded_file is not None:
    content_img = Image.open(content_uploaded_file)

    test_1 = col1.image(content_img,
                        caption='test',
                        channels="RGB",
                        output_format='PNG',
                        use_column_width=True)
    content_img=content_img.resize((299, 299), Image.ANTIALIAS)

    # convert image to bytes
    img_byte_arr = io.BytesIO()
    content_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    with open("image.jpg", "wb") as f:
        f.write(img_byte_arr)

st.set_option("deprecation.showfileUploaderEncoding", False)
style_uploaded_file = col2.file_uploader("Choose an style image:",
                                         type=["png", "jpeg",'jpg'])

if style_uploaded_file is not None:
    style_img = Image.open(style_uploaded_file)

    test_1 = col2.image(style_img,
                        caption='test',
                        channels="RGB",
                        output_format='PNG',
                        use_column_width=True)
    style_img=style_img.resize((299, 299), Image.ANTIALIAS)

    # convert image to bytes
    img_byte_arr2 = io.BytesIO()
    style_img.save(img_byte_arr2, format='PNG')
    img_byte_arr2 = img_byte_arr2.getvalue()

    with open("image.jpg", "wb") as f:
        f.write(img_byte_arr2)


    # api call
    url = "http://127.0.0.1:8000/create"
    files = {"content": img_byte_arr, "style": img_byte_arr2}

    with requests.Session() as s:
        response = s.post(url,files=files)

    if response.status_code == 200:
        resp = response.json()
        result=np.array(resp["result"]).reshape(resp["shape"])
        st.image(Image.fromarray((result * 255).astype(np.uint8)))

    else:
        resp = response.json()
        resp
        "😬 api error 🤖"
