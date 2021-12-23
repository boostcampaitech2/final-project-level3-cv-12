# Sketch2Idol: 나만의 여자 아이돌 만들기

본 프로젝트는 GAN를 이용해 손으로 그린 스케치를 아이돌 이미지로 변환하는 Image To Image Translation을 주제로 합니다.

![project_figure](https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/main2.png?raw=true)

## Data

- airflow: 데이터 파이프라인 자동화
  - webserver_config.py: airflow 웹 서버 설정 파일
  - airflow.cfg: airflow 스케줄러 설정 파일
  - airflow.db
  - airflow-webserver.pid
  - dags
    - data_pipeline.py: 데이터 파이프라인 자동화 설정 파일

- dockers
  - google_api_dir: 얼굴 추출 및 눈, 코, 입 포인트 json 파일 생성
    - Dockerfile
    - google_vision_api.py:
    - init.sh
    - remover.py: 이미지의 메타데이터를 삭제하여 전처리 안정화
    - rename.py: 이미지 이름 순서대로 할당
    - requirements.txt
  - photocopy: 이미지 스케치화
    - Dockerfile
    - alignment.py: 모든 이미지의 눈, 입을 기준으로 이미지 재배열
    - init.sh
    - photocopy.py: OpenCV를 통한 스케치 이미지 생성
    - requirements.txt
    - sketch.py: 스케치 이미지 가공
  - super_resolution: 이미지 화질 향상 및 통일
    - Dockerfile
    - init.sh
    - requirements.txt
  - mask.sh: 데이터 파이프 라인 수행 쉘 스크립트
  - image_files : 이미지 관리 및 저장 폴더

- upload_server
  - number : 이미지 중복 방지를 위한 넘버링 파일
  - server.py : 사용자로부터 이미지를 수신하는 서버
  - poetry.lock
  - pyproject.toml

## Model



## Folder structure
## Authors

박진한|유형진|이양재|임채영|최태종|한재현
:-:|:-:|:-:|:-:|:-:|:-:
<img src='https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/member/1.png?raw=true' height=80 width=80px></img>|<img src='https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/member/2.png?raw=true' height=80 width=80px></img>|<img src='https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/member/3.png?raw=true' height=80 width=80px></img>|<img src='https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/member/1.png?raw=true' height=80 width=80px></img>|<img src='https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/member/2.png?raw=true' height=80 width=80px></img>|<img src='https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/member/3.png?raw=true' height=80 width=80px></img>
[Github](https://github.com/jinhan814)|[Github](https://github.com/jinhan814)|[Github](https://github.com/jinhan814)|[Github](https://github.com/jinhan814)|[Github](https://github.com/jinhan814)|[Github](https://github.com/jinhan814)

## reference

- [DeepFaceDrawing: Deep Generation of Face Images from Sketches](http://geometrylearning.com/paper/DeepFaceDrawing.pdf)
- [Nonlinear Dimensionality Reduction by Locally Linear Embedding](https://www.robots.ox.ac.uk/~az/lectures/ml/lle.pdf)
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation](https://arxiv.org/abs/1907.10830)