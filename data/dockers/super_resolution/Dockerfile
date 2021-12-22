FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

WORKDIR /home/myamya/project/

COPY ./requirements.txt .
COPY ./init.sh .
run apt-get update
run apt-get install -y git
run git clone https://github.com/xinntao/Real-ESRGAN.git
run pip install -r requirements.txt
WORKDIR /home/myamya/project/Real-ESRGAN
run python setup.py develop
run apt-get install -y wget
run apt-get install -y libgl1-mesa-glx
run apt-get install -y libglib2.0-0
run wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
