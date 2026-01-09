🐦 Twitter Sentiment Analysis using BiLSTM
📌 Overview

This project implements an end-to-end Twitter Sentiment Analysis system using a Bidirectional LSTM (BiLSTM) deep learning model.
It classifies text into Negative, Neutral, or Positive sentiment and includes model training as well as a Streamlit-based web application for real-time predictions.

✨ Key Features

✔ Deep Learning–based sentiment classification (BiLSTM)
✔ Robust text preprocessing & cleaning pipeline
✔ Handles class imbalance using class weights
✔ Real-time sentiment prediction with Streamlit
✔ Clean, modular, and deployment-ready codebase

🏗️ Project Structure
twitter_sentiment_project/
├── app/
│   └── twitter_sentiment_app.py        # Streamlit application
├── models/
│   ├── sentiment_model.h5              # Trained BiLSTM model
│   └── tokenizer.joblib                # Saved tokenizer
├── data/
│   └── Twitter_Data.csv                # Dataset
├── notebooks/
│   └── modeldeployment.ipynb           # Experiments & notes
├── twitter_sentiment_main.py            # Model training script
├── requirements.txt
├── .gitignore
└── README.md

📊 Dataset

Name: Twitter_Data.csv

Text Column: text

Target Column: sentiment

Classes:

🔴 Negative

⚪ Neutral

🟢 Positive

The dataset is preprocessed to remove URLs, mentions, special characters, and unnecessary whitespace.

🧠 Model Architecture

The sentiment classifier is built using the following architecture:

🔤 Embedding Layer

🔁 SpatialDropout1D

🔄 Bidirectional LSTM

🧱 Dense Layer (ReLU)

🎯 Dropout

📤 Softmax Output Layer

Loss Function: Sparse Categorical Crossentropy
Optimizer: Adam

🧪 Model Training Pipeline

The training workflow includes:
1️⃣ Data cleaning and preprocessing
2️⃣ Tokenization and sequence padding
3️⃣ Stratified train–test split
4️⃣ Handling class imbalance using class weights
5️⃣ Early stopping to prevent overfitting
6️⃣ Model evaluation using:

📄 Classification Report

🔢 Confusion Matrix

Saved Artifacts:

sentiment_model.h5

tokenizer.joblib

🌐 Streamlit Web Application

The Streamlit app allows users to:
📝 Enter any tweet or short text
⚡ Get instant sentiment predictions
📊 View class-wise probability scores
