import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from artia.NST_model import img_to_tensor, tensor_to_image
import tensorflow_hub as hub
import tensorflow as tf
import requests
import io

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

    #img = content_img
    #img.resize((299, 299), Image.ANTIALIAS)
    #content_img = np.array(img)[:, :, 0:3].astype(float) / 255

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

    # img = style_img
    # img.resize((299, 299), Image.ANTIALIAS)
    # style_img = np.array(img)[:, :, 0:3].astype(float) / 255.

    # convert image to bytes
    img_byte_arr2 = io.BytesIO()
    style_img.save(img_byte_arr2, format='PNG')
    img_byte_arr2 = img_byte_arr2.getvalue()

    with open("image.jpg", "wb") as f:
        f.write(img_byte_arr2)

    # model='tensor'
    # if model=='local':
    #     st.image(tensor_to_image(content_img, style_img))
    # else:

    #     hub_model = hub.load(
    #         'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    #     stylized_image = hub_model(tf.constant(img_to_tensor(content_img)),
    #                                tf.constant(img_to_tensor(style_img)))[0]

    #     tensor = stylized_image * 255
    #     tensor = np.array(tensor, dtype=np.uint8)
    #     if np.ndim(tensor) > 3:
    #         assert tensor.shape[0] == 1
    #         tensor = tensor[0]

    #     st.image(Image.fromarray(tensor))


    # api call
    url = "http://127.0.0.1:8000/create"
    files = {"content": img_byte_arr, "style": img_byte_arr2}

    response = requests.post(url, files=files)

    if response.status_code == 200:
        resp = response.json()
        resp
    else:
        print(files)
        resp = response.json()
        resp
        "😬 api error 🤖"
