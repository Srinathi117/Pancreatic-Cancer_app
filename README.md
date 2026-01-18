# Pancreatic Cancer Prediction using CNN + GNN with AI Chatbot Assistant

## Overview

Pancreatic cancer is a deadly disease with a low survival rate, and early detection is critical for improving patient outcomes. This project aims to predict the presence of pancreatic cancer from CT/MRI scan images using an advanced deep learning approach (GNN + Graph Neural Network). A ResNet18 backbone is used for feature extraction and a Graph Convolutional Network (GCN) is used for classification. The system is deployed as a Flask web application with an integrated AI Chatbot Assistant (Gemini API) for general awareness and user guidance.

## Dataset
The dataset used for this project contains pancreatic CT/MRI scan images with corresponding labels.

 Note: Any information such as cancer stage (if present in the dataset) is not used for prediction. The main goal is to support early insight based on imaging data, not post-diagnosis staging.

 ## Requirements

 - Python
 - Libraries and dependencies:
     - Flask
     - Torch
     - Torchvision
  - Torch-geometric
  - Pillow
  - Google-generativeai (Gemini)
## Usage

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/Srinathi117/Pancreatic-Cancer_app.git


## Data Preprocessing

Explain the preprocessing steps performed on CT/MRI images, such as:

   - Resizing images to 224×224
   - Converting to tensor
   - Normalization using ImageNet mean and std
   - Augmentation like rotation/flip if used during training

## Model Training

Provide details about the hybrid model:
 
   - CNN Backbone: ResNet18 (ImageNet pretrained)
   - Graph Model: GCNConv layers (GNN)
   - Model checkpoint: pancreas_gnn.pth
     
Training details:

   - Optimizer, learning rate, epochs
   - Loss function (e.g., CrossEntropyLoss)
   - Hyperparameters: hidden channels, dropout, etc.

## Evaluation

Discuss the metrics used to evaluate the model performance such as:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix

## Results

Summarize the findings and results of pancreatic cancer prediction using the CNN + GNN model, including:
  - Performance metrics
  - Confidence output
  - Model behavior on cancer vs non-cancer cases

## AI Chatbot Assistant

 The web app includes a Gemini-powered AI Assistant that can:
   - Explain how to use the website
   - Provide general awareness about pancreatic cancer
   - Answer basic questions about CT/MRI imaging and AI models
The chatbot does not provide medical diagnosis or treatment advice.

## Acknowledgments

Any acknowledgments or credits for:
  - Dataset sources
  - PyTorch / Torch Geometric
  - Flask
  - Gemini API

## Disclaimer

This project is for educational and research purposes only. It should not be used as a substitute for medical advice or diagnosis. Consult a healthcare professional for any concerns related to pancreatic health.
