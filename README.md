# AI Email Safety Assistant

AI Email Safety Assistant is a lightweight machine-learning web application that classifies messages as HAM (legitimate) or SPAM and provides a risk level and confidence score.

## Problem

Spam and suspicious messages can contain unwanted advertisements, fake rewards, misleading information, or potentially harmful links. This project provides a simple way to screen message content before interacting with it.

## Features

- Spam/Ham classification
- High/Low risk indication
- Confidence score
- Web-based interface
- Flask API for real-time prediction
- Local model inference

## Technologies Used

- HTML
- CSS
- JavaScript
- Python
- Flask
- Scikit-learn
- TF-IDF
- Logistic Regression

## Dataset

The project uses the UCI SMS Spam Collection dataset containing 5,572 labeled messages.

## Model Performance

The trained model achieved approximately **97% accuracy on the held-out test dataset**.

This result is specific to the dataset and test split used in the prototype and does not represent guaranteed real-world accuracy.

## Project Structure

```text
backend/
    app.py
    test_model.py

dataset/
    emails.csv

frontend/
    index.html

model/
    spam_model.pkl
    tfidf_vectorizer.pkl

training/
    convert_dataset.py
    train_model.py
