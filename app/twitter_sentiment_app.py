"""
twitter_sentiment_app.py

Streamlit app for Twitter sentiment analysis using:
- sentiment_model.h5
- tokenizer.joblib

Make sure these files are in the same folder before running:

    streamlit run twitter_sentiment_app.py
"""
import os
print("Current working dir:", os.getcwd())

import re
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load

# Paths must match what you used in training
MODEL_PATH = "models/sentiment_model.h5"
TOKENIZER_PATH = "models/tokenizer.joblib"

# These must match training configuration
MAX_SEQUENCE_LENGTH = 40
ID_TO_LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}


# ---------------------------------------------------------
# Cleaning function (must match training)
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Same cleaning as in training script:

    - Remove URLs
    - Remove @mentions
    - Keep letters, spaces, '!' and '?'
    - Lowercase
    - Collapse multiple spaces
    """
    text = str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"[^a-zA-Z\s!?]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------
# Load model and tokenizer (cached)
# ---------------------------------------------------------

@st.cache_resource
def load_model_and_tokenizer():
    """
    Load and cache the trained Keras model and tokenizer.
    """
    model = tf.keras.models.load_model(MODEL_PATH)
    tokenizer = load(TOKENIZER_PATH)
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()


# ---------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------

def predict_sentiment(raw_text: str):
    """
    Clean the input text, convert to padded sequence, and get model prediction.

    Returns:
        pred_label (str): "Negative" / "Neutral" / "Positive"
        proba (np.ndarray): probability distribution over classes
    """
    cleaned = clean_text(raw_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    proba = model.predict(padded)[0]      # shape (3,)
    pred_id = int(np.argmax(proba))
    pred_label = ID_TO_LABEL[pred_id]

    return pred_label, proba


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.set_page_config(page_title="Twitter Sentiment Analysis", layout="centered")

st.title("Twitter Sentiment Analysis")
st.write(
    "This app uses a BiLSTM model trained on a Twitter dataset "
    "to classify text as **Negative**, **Neutral**, or **Positive**."
)

user_input = st.text_area(
    "Enter a tweet or any short text:",
    height=150,
    placeholder="Type something like: I love this new phone!"
)

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text for analysis.")
    else:
        label, proba = predict_sentiment(user_input)

        st.subheader("Prediction")
        st.markdown(f"**Sentiment:** {label}")

        st.subheader("Class probabilities")
        st.write(
            {
                "Negative": float(proba[0]),
                "Neutral": float(proba[1]),
                "Positive": float(proba[2]),
            }
        )