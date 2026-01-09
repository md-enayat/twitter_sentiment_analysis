"""
twitter_sentiment_main.py

Train a BiLSTM sentiment model on Twitter_Data.csv and save:
- sentiment_model.h5
- tokenizer.joblib

Pipeline:
1. Load and clean data
2. Train/test split
3. Tokenize and pad sequences
4. Build BiLSTM model
5. Train with EarlyStopping and class weights
6. Evaluate (classification_report + confusion_matrix)
7. Save model and tokenizer for deployment (Streamlit app)
"""

# ---------------------------------------------------------
# Imports and configuration
# ---------------------------------------------------------

import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from joblib import dump

# File paths
DATA_PATH = "Twitter_Data.csv"        # input dataset
MODEL_PATH = "sentiment_model.h5"     # output model
TOKENIZER_PATH = "tokenizer.joblib"   # output tokenizer

# Reproducibility
RANDOM_STATE = 42

# Model & text hyperparameters
MAX_NUM_WORDS = 20000         # vocabulary size for Tokenizer
MAX_SEQUENCE_LENGTH = 40      # max tokens per tweet
EMBEDDING_DIM = 100           # embedding size
EPOCHS = 15                   # maximum epochs (with EarlyStopping)
BATCH_SIZE = 32               # batch size

# Label mappings
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
NUM_CLASSES = len(LABEL_TO_ID)


# ---------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Simple tweet cleaning used for BOTH training and inference.

    Steps:
    - Ensure input is a string
    - Remove URLs
    - Remove @mentions
    - Keep only letters, spaces, '!' and '?'
    - Lowercase
    - Collapse multiple spaces
    """
    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)

    # Keep letters, spaces, and basic sentiment punctuation (! and ?)
    text = re.sub(r"[^a-zA-Z\s!?]", " ", text)

    # Lowercase
    text = text.lower()

    # Collapse multiple whitespaces and strip
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    Load Twitter_Data.csv and produce a cleaned DataFrame with:
        - clean_text : cleaned tweet text
        - sentiment  : original sentiment label (string)
        - label_id   : numeric label (0, 1, 2)
    """

    # Load raw data
    DATA_PATH = r'D:\Boston Institute of Analytics (BIA)\Project DL\twitter\data\Twitter_Data.csv'
    df_raw = pd.read_csv(DATA_PATH)

    # Drop rows where text or sentiment is missing
    df = df_raw.dropna(subset=["text", "sentiment"]).copy()

    # Apply text cleaning
    df["clean_text"] = df["text"].apply(clean_text)

    # Drop very short cleaned tweets (length <= 2)
    df = df[df["clean_text"].str.len() > 2]

    # Map sentiment labels to integer IDs
    df["label_id"] = df["sentiment"].map(LABEL_TO_ID)

    # Drop any rows with unmapped labels (should not happen)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    return df


# ---------------------------------------------------------
# Model building
# ---------------------------------------------------------

def build_model(
    max_num_words: int,
    max_sequence_length: int,
    embedding_dim: int,
    num_classes: int
):
    """
    Build a BiLSTM-based text classification model:

    Embedding -> SpatialDropout1D -> BiLSTM -> Dense(64, relu) -> Dropout -> Dense(num_classes, softmax)
    """
    model = models.Sequential([
        layers.Embedding(
            input_dim=max_num_words,
            output_dim=embedding_dim,
            input_length=max_sequence_length
        ),
        layers.SpatialDropout1D(0.1),
        layers.Bidirectional(layers.LSTM(128)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    # Using sparse_categorical_crossentropy because labels are int IDs (0, 1, 2)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ---------------------------------------------------------
# Main training flow
# ---------------------------------------------------------

def main():
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(DATA_PATH)
    print(f"Rows after cleaning: {len(df)}")

    print("\nLabel distribution (sentiment):")
    print(df["sentiment"].value_counts())

    # Features and labels
    X = df["clean_text"].values
    y = df["label_id"].values

    # 2. Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # 3. Tokenize & pad
    print("\nFitting tokenizer and converting to padded sequences...")
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(
        X_train_seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )
    X_test_pad = pad_sequences(
        X_test_seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    # Ensure numeric arrays
    X_train_pad = np.asarray(X_train_pad, dtype="int32")
    X_test_pad = np.asarray(X_test_pad, dtype="int32")
    y_train = np.asarray(y_train, dtype="int32")
    y_test = np.asarray(y_test, dtype="int32")

    # 4. Build model
    print("\nBuilding BiLSTM model...")
    model = build_model(
        max_num_words=MAX_NUM_WORDS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES
    )
    model.summary(print_fn=lambda x: print(x))

    # 5. Compute class weights (helps with under-predicted classes)
    print("\nComputing class weights...")
    classes = np.array([0, 1, 2])  # negative, neutral, positive
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights_array)}
    print("Class weights:", class_weight_dict)

    # Early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    # 6. Train model
    print("\nTraining model...")
    history = model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_test_pad, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
        class_weight=class_weight_dict
    )

    # 7. Evaluate
    print("\nEvaluating on test set...")
    y_pred_prob = model.predict(X_test_pad)
    y_pred = np.argmax(y_pred_prob, axis=1)

    target_names = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    print("Confusion matrix (rows = true, cols = pred):")
    print(confusion_matrix(y_test, y_pred))

    # 8. Save model & tokenizer
    print(f"\nSaving model to {MODEL_PATH} ...")
    model.save(MODEL_PATH)

    print(f"Saving tokenizer to {TOKENIZER_PATH} ...")
    dump(tokenizer, TOKENIZER_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()