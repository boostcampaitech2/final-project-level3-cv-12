# Sketch2Idol: 나만의 여자 아이돌 만들기

## Introduction

본 프로젝트는 GAN를 이용해 손으로 그린 스케치를 아이돌 이미지로 변환하는 Image To Image Translation을 주제로 합니다.

![project_pipeline](https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/main.png?raw=true)

## Dataset

|Data|데이터 수|Train 데이터 수|Val 데이터 수|세부사항|
|:-:|:-:|:-:|:-:|:-:|
|1|2449|2429|20|web crawling, 여자 아이돌|
|2|1669|1669|0|aihub, 일반인|

#### Info

- 한국 여성 얼굴 이미지
- 이미지-스케치 pair
- face alignment 적용
- (512, 512) 해상도

## Model

![project_pipeline](https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/pipeline.png?raw=true)

- Encoder : (1, 512, 512) -> (512)
- Decoder : (512) -> (1, 512, 512)
- Pix2Pix : (1, 512, 512) -> (3, 512, 512)

## Folder structure

```
WIP
```

## Authors

|박진한|유형진|이양재|임채영|최태종|한재현|
|:-:|:-:|:-:|:-:|:-:|:-:|
|<img src='https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/member/1.png?raw=true' height=80 width=80px></img>|<img src='https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/member/2.png?raw=true' height=80 width=80px></img>|<img src='https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/member/3.png?raw=true' height=80 width=80px></img>|<img src='https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/member/1.png?raw=true' height=80 width=80px></img>|<img src='https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/member/2.png?raw=true' height=80 width=80px></img>|<img src='https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/member/3.png?raw=true' height=80 width=80px></img>|
|[Github](https://github.com/jinhan814)|[Github](https://github.com/jinhan814)|[Github](https://github.com/jinhan814)|[Github](https://github.com/jinhan814)|[Github](https://github.com/jinhan814)|[Github](https://github.com/jinhan814)|

## reference

- [DeepFaceDrawing: Deep Generation of Face Images from Sketches](http://geometrylearning.com/paper/DeepFaceDrawing.pdf)
- [Nonlinear Dimensionality Reduction by Locally Linear Embedding](https://www.robots.ox.ac.uk/~az/lectures/ml/lle.pdf)
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation](https://arxiv.org/abs/1907.10830)