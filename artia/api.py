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



@app.post("/create")
async def create(content, style):

    print("\nreceived file:")
    print(type(content))
    #print(content_img)

    #image_path = "image_api.png"

    # write file to disk
    #with open(image_path, "wb") as f:
    #    f.write(file)

    # model -> pred

    return dict(pred=True)
