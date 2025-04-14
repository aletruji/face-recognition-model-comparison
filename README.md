"""
## Comparison of Classical and Modern Machine Learning Models for Face Recognition

This project implements and compares classical face recognition methods (Eigenfaces, Fisherfaces, LBPH)
with modern deep learning models (FaceNet, ArcFace, Dlib-ResNet).
The goal is to evaluate all six models under identical conditions using a unified dataset and standardized evaluation metrics.

## Objective

The aim is to build a modular comparison framework that applies each model to the same face recognition tasks.
The performance of each model is measured using classification metrics such as accuracy, precision, recall, F1-score, and inference time.
The results provide a foundation for analysis in a bachelor’s thesis.

Used lfw-Dataset

-https://archive.org/download/lfw-dataset


Added Dlib pretrained model files:

- `shape_predictor_68_face_landmarks.dat`  
  → used for face alignment (68-point facial landmarks)  
  → Source: https://github.com/davisking/dlib-models

- `dlib_face_recognition_resnet_model_v1.dat`  
  → used for computing 128D face embeddings  
  → Source: https://github.com/davisking/dlib-models

"""
