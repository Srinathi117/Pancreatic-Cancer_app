# Pancreatic Cancer Prediction using CNN + GNN (GCN) with AI Chatbot Assistant

## Overview

Pancreatic cancer is a deadly disease with a low survival rate, and early detection is critical for improving patient outcomes. This project aims to predict the presence of pancreatic cancer from CT/MRI scan images using an advanced deep learning approach (CNN + Graph Neural Network). A ResNet18 backbone is used for feature extraction and a Graph Convolutional Network (GCN) is used for classification. The system is deployed as a Flask web application with an integrated AI Chatbot Assistant (Gemini API) for general awareness and user guidance.

## The dataset used for this project contains pancreatic CT/MRI scan images with corresponding labels.

 Note: Any information such as cancer stage (if present in the dataset) is not used for prediction. The main goal is to support early insight based on imaging data, not post-diagnosis staging.

 ## Requirements

 - Python
 - Libraries and dependencies:
     - Flask
     - torch,
     - torchvision
  - torch-geometric
  - Pillow
  - google-generativeai (Gemini)
