from fastapi import FastAPI, File, UploadFile
from typing import List
import os
import uvicorn
import fileinput

app = FastAPI()
allowd_ext = ['.jpg','.jpeg','.png', '.bmp']
@app.get("/")
def read_root():
  return { "Hello": "World" }
@app.post("/files/")
async def create_files(files: List[bytes] = File(...)):
    return {"file_sizes": [len(file) for file in files]}
@app.post("/uploadfiles")
async def create_upload_files(files: List[UploadFile] = File(...)):
    UPLOAD_DIRECTORY = "/home/myamya/project/image_files/images/"
    for file in files:
        contents = await file.read()
        ext = os.path.splitext(file.filename)[1]
        if ext in allowd_ext: 
            with open(os.path.join(os.getcwd(),'number'), 'r+') as fn:
                number =fn.read()
                fn.seek(0)
                number2 =int(number)
                number2+=1
                fn.write(str(number2))
                file.filename = str(number2) + ext
            with open(os.path.join(UPLOAD_DIRECTORY, file.filename), "wb") as fp:

                fp.write(contents)
            print(file.filename)
        else:
            print('wrong format')
    return {"filenames": [file.filename for file in files]} 

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
