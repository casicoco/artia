from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from artia.NST_model import tensor_to_image



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

    # example of how you can save the file
    with open(f"{content.filename}", "wb") as f:
        f.write(content_img)

    style.filename = "style"
    style_img = await style.read()  # <-- Important!

    # example of how you can save the file
    with open(f"{style.filename}", "wb") as f:
        f.write(style_img)


    return {"content": content.filename,"style":style.filename}

"""
@app.get("/images/")
async def read_random_file():

    # get a random file from the image directory
    files = os.listdir(IMAGEDIR)
    random_index = randint(0, len(files) - 1)

    path = f"{IMAGEDIR}{files[random_index]}"
    
    # notice you can use FileResponse now because it expects a path
    return FileResponse(path)"""
