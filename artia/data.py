import numpy as np
import pandas as pd
from PIL.Image import open,
import matplotlib.pyplot as plt

import tensorflow as tf

from os.path import join,dirname

'''Data conversion from image to tensor object'''

def load_img(default=True, content_image=None, style_image=None):
    '''data_path to raw_data to extract content & style images'''

    content_data_path = join(dirname(dirname(__file__)),'raw_data','content.jpeg')
    style_data_path = join(dirname(dirname(__file__)),'raw_data','style.png')

    if content_image:
        content_data_path=content_image
    if style_image:
        style_data_path=content_image

    return {'content':content_data_path,'style':style_data_path}


def convert_to_tensor(path_to_img, content_image=None, style_image=None):
    '''Loading image and convertion into a normalized tensor image'''

    if not content_image:
        img = tf.io.read_file(path_to_img)             #convert img into a tensor object
        img = tf.io.decode_image(img, channels=3,dtype=tf.float32)      #convert img into image of shape (pixels height, pixels width & RGB)
        img = tf.image.convert_image_dtype(img, tf.float32)    #normalization of image float

    return img
