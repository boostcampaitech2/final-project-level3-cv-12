import os
import cv2
import numpy as np
import torch
import re
import base64
from typing import List, Union, Optional, Dict, Any
from manifold.manifold import KNN, ConstrainedLeastSquareSolver
from model.model import get_encoder, get_decoder, get_generator, get_knn, inference, save_image
from fastapi import FastAPI, UploadFile, File, Body,Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

app = FastAPI()
loading=False
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

@app.on_event('startup')
def load_models():
	encoder_path   = '/opt/ml/backendv3/model_save/encoder_all_260_1216_1424.pth'
	decoder_path   = '/opt/ml/backendv3/model_save/decoder_all_260_1216_1424.pth'
	generator_path = '/opt/ml/backendv3/model_save/sketch2image_params_0096000.pt'
	fv_json_path   = '/opt/ml/backendv3/model_save/fv_all_train.json'

	global device, encoder, decoder, generator, knn, least_square
	use_cuda = torch.cuda.is_available()
	device = torch.device('cuda' if use_cuda else 'cpu')
	encoder = get_encoder(encoder_path, 'all', device)
	decoder = get_decoder(decoder_path, 'all', device)
	generator = get_generator(generator_path, device)
	generator.eval()
	knn = get_knn(fv_json_path)
	least_square = ConstrainedLeastSquareSolver()

	# filename='/opt/ml/backendv3/some_image.png'
	# sketch_img = 1 - cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(float) / 255
	# inferenced_img = inference(sketch_img, encoder, decoder, generator, knn, least_square, device, 0.4, 10)
	

@app.get("/")
def hello_world():
   return {"hello": "world"}

@app.post("/")
async def test(request: Request):
	loading = True
	item = await request.json()
	data = item['image']
	imgstring = re.sub('^data:image/.+;base64,', '', data)
	imgdata = base64.b64decode(imgstring)
	filename = 'some_image.png'  # I assume you have a way of picking unique filenames
	with open(filename, 'wb') as file:
		file.write(imgdata)

	t = item['T']
	k = 10
	sketch_img = 1 - cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(float) / 255
	inferenced_img = inference(sketch_img, encoder, decoder, generator, knn, least_square, device, t, k)
	return {"hello": "world"}

@app.get("/output.png")
def image():
   while (loading):
      sleep(0.05)
   return FileResponse("./output.png")
