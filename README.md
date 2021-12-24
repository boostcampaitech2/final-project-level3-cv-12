# Sketch2Idol: 나만의 여자 아이돌 만들기

## Introduction

본 프로젝트는 GAN를 이용해 손으로 그린 스케치를 아이돌 이미지로 변환하는 Image To Image Translation을 다룹니다.

![project_pipeline](https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/main.png?raw=true)

## Dataset

|Data|데이터 수|Train 데이터 수|Val 데이터 수|세부사항|
|:-:|:-:|:-:|:-:|:-:|
|1|2449|2429|20|web crawling, 여자 아이돌|
|2|1669|1669|0|[aihub](https://aihub.or.kr/aidata/27716), 일반인|

학습에는 크롤링으로 모은 2449장의 여자 아이돌 이미지와 AIHub 한국인 감정인식을 위한 복합 영상 데이터셋의 1669장의 이미지를 활용했습니다.

각 이미지는 resize, super resolution를 거쳐 (512, 512) 해상도로 변환한 뒤, OpenCV, Learning to Simplify 코드를 이용해 스케치로 변환됩니다. 이후 이미지-스케치 pair는 google API를 이용해 눈, 입의 좌표를 정규화하는 face alignment를 적용해 저장됩니다.

데이터 중 정면에서 고개가 10도 이상 돌아간 사진, 마이크, 손 등으로 얼굴 일부를 가린 사진, 얼굴에 그림자가 있거나 조명이 너무 강한 사진은 제거했습니다.

## Model

![project_pipeline](https://github.com/boostcampaitech2/final-project-level3-cv-12/blob/main/sample_image/pipeline.png?raw=true)

전체적인 파이프라인은 <sup>1)</sup>AutoEncoder를 이용해 512*512 크기의 스케치를 512차원의 feature vector로 encoding한 뒤 <sup>2)</sup>데이터셋의 스케치들의 feature vector와 가중 평균을 낸 뒤 decoding해서 실제 사진과 유사한 스케치를 얻고, <sup>3)</sup>이렇게 얻은 스케치를 Pix2Pix를 이용해 실제 사진으로 변환하는 구조로 이루어져있습니다.

## Folder structure

```
WIP
```

## Authors

|박진한|유형진|이양재|임채영|최태종|한재현|
|:-:|:-:|:-:|:-:|:-:|:-:|
|<img src='https://avatars.githubusercontent.com/u/77492810?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/84146296?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/19571027?v=4?raw=true' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/63492979?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/87696070?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/53294402?v=4' height=80 width=80px></img>|
|[Github](https://github.com/jinhan814)|[Github](https://github.com/tkdlqh2)|[Github](https://github.com/yayaja11)|[Github](https://github.com/chay116)|[Github](https://github.com/ssail09)|[Github](https://github.com/eric9687)|

## reference

- [DeepFaceDrawing: Deep Generation of Face Images from Sketches](http://geometrylearning.com/paper/DeepFaceDrawing.pdf)
- [Learning to simplify: fully convolutional networks for rough sketch cleanup](http://www.f.waseda.jp/hfs/SimoSerraSIGGRAPH2016.pdf)
- [Nonlinear Dimensionality Reduction by Locally Linear Embedding](https://www.robots.ox.ac.uk/~az/lectures/ml/lle.pdf)
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation](https://arxiv.org/abs/1907.10830)