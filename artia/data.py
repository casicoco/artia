import numpy as np
import pandas as pd

from tensorflow import XX

from os.path import join,dirname

'''Data conversion from image to tensor object'''

#data_path to raw_data to extract content & style images
content_data_path = join(dirname(dirname(__file__)),'raw_data','content.jpeg')
style_data_path = join(dirname(dirname(__file__)),'raw_data','style.png')
