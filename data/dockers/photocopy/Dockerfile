FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

WORKDIR /home/myamya/project/

COPY ./requirements.txt .
COPY ./sketch.py . 
COPY ./init.sh .
run pip install --upgrade pip
run apt-get update
run apt-get install -y libgl1-mesa-glx
run apt-get install -y libglib2.0-0
run pip install -r requirements.txt
run git clone https://github.com/bobbens/sketch_simplification.git
run apt-get install -y wget
run wget --no-check-certificate https://esslab.jp/~ess/data/sketch_gan.t7


