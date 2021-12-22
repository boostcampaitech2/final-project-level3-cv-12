from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import base64
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any
import torch
from datetime import datetime
from model.model import get_encoder, get_decoder,get_knn,get_generator, inference
from manifold.manifold import KNN, ConstrainedLeastSquareSolver
from fastapi import Body
import re

app = FastAPI()

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

@app.on_event('startup')
def load_models():
	encoder_file_path  = '/opt/ml/backend/project/_DB/encoder_all_260_1216_1424.pth'
	decoder_file_path  = '/opt/ml/backend/project/_DB/decoder_all_260_1216_1424.pth'
	generator_pth_path = '/opt/ml/backend/project/_DB/generator/generator_tj.pth'
	fv_json_path       = '/opt/ml/backend/project/_DB/fv_all_train.json'

	use_cuda = torch.cuda.is_available()
	global device
	device = torch.device('cuda' if use_cuda else 'cpu')
	global encoder
	encoder = get_encoder(encoder_file_path, 'all', device)
	global decoder
	decoder = get_decoder(decoder_file_path, 'all', device)
	global generator
	generator = get_generator(generator_pth_path, device)
	global knn
	knn = get_knn(fv_json_path)
	encoder.to(device)
	decoder.to(device)
	generator.to(device)

@app.get("/")
def hello_world():
	return {"hello": "world"}

@app.post("/")
async def test(data: str = Body(...)):
	imgstring = re.sub('^data:image/.+;base64,', '', data)
	imgdata = base64.b64decode(imgstring)
	filename = 'some_image.png'  # I assume you have a way of picking unique filenames
	with open(filename, 'wb') as file:
		file.write(imgdata)
	least_square = ConstrainedLeastSquareSolver()
	t = 0.4
	k = 10
	inference(filename, encoder,decoder,generator,knn,least_square, device, t, k)
	return {"hello": "world"}

