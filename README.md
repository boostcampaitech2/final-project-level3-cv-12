# Project

## Goals

- Implementing DeepFaceDrawing Paper
  - http://geometrylearning.com/DeepFaceDrawing/
  - http://www.geometrylearning.com/paper/DeepFaceDrawing.pdf
  - http://geometrylearning.com/paper/DeepFaceDrawing-supple.pdf

- freehand sketch to realistic face image

## Folder Structure

- Moudle : train과 inference에 필요한 여러 가지 파일들

## Model Structure

Part 1. *Component Embedding Module*(CE)

- Keywords
  - auto encoder(AE)
  - nonlinear dimensionality reduction, manifold learning, feature extraction
  - Locally Linear Embedding(LLE) (https://arxiv.org/abs/1406.2661)
  - k-nearest neighbor(KNN)
    - O(n) naive algorithm
    - O(n) preprocessing, O(logn) query algorithm using KD-Tree
    - hybrid algorithm using KD-Tree and naive algorithm
  - interpolation, constrained least-squares problem

- input / output
  - input : (512 x 512) sketch image
  - output : (512) feature vector, (512 x 512) decoded image

- process
  - train 5 different AE (left-eye, right-eye, nose, mouth, remainder)
  - Decode (H x W) input image to (512) vector
  - Find k-nearest neighbor
  - interpolate KNN by using constrained least-square problem

Part 2. *Feature Mapping Module*(FM)

- Keywords
  - decoder
  - feature vector to feature map
  - concatnation

- input / output
  - input : (512) feature vector
  - output : (512 x 512 x 32) feature map

Part 3. *Image Synthesis Module*(IS)

- Keywords
  - GAN(https://arxiv.org/abs/1406.2661)
  - conditional GAN(https://arxiv.org/abs/1411.1784)
  - pix2pixHD (https://arxiv.org/abs/1611.07004, https://arxiv.org/abs/1711.11585)

- input / output
  - input : (512 x 512 x 32) feature map
  - output : (512 x 512) synthesized image