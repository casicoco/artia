from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from NST_model import tensor_to_image

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


@app.get("/create")
def create(content_img, style_img):
    return tensor_to_image(content_img, style_img)
