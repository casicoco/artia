from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from artia.NST_model import tensor_to_image
import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image
import io



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"])  #Allows all origins, all methods and all headers


@app.get("/")
def index():
    return {"greeting": "Hello world"}



@app.post("/create/")
async def create_upload_file(content: UploadFile = File(...),style: UploadFile = File(...)):
    """
    Takes in the 2 pictures (content and style) and saves them locally for now
    """
    content.filename = "content"
    content_img = await content.read()  # <-- Important!

    content_img=tf.io.decode_image(content_img,
                           channels=3,
                           dtype=tf.dtypes.float32,
                           name=None,
                           expand_animations=True)

    style.filename = "style"
    style_img = await style.read()  # <-- Important!

    style_img = tf.io.decode_image(style_img,
                                   channels=3,
                                   dtype=tf.dtypes.float32,
                                   name=None,
                                   expand_animations=True)

    # example of how you can save the file
    # with open(f"{style.filename}", "wb") as f:
    #     f.write(style_img)

    tensor_result=tensor_to_image(content_img, style_img)
    shape=tensor_result.shape
    np_result=tensor_result.reshape([-1])
    np_result=np_result.tolist()
    # print(len(np_result))
    # print(type(np_result))

    return {"result": np_result, "shape":shape}
