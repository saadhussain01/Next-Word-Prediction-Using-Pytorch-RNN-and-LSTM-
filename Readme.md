### 🧠 Next Word Prediction using RNN (PyTorch)

This project implements a Next Word Predictor built with Recurrent Neural Networks (RNNs) using PyTorch. The model is trained on a custom dataset of 100 unique Question-Answer pairs (QA dataset) in CSV format. The goal is to predict the next possible word in a given sequence, demonstrating how sequential models capture context and language structure.

## 🚀 Project Overview

Language modeling is one of the fundamental problems in Natural Language Processing (NLP).
This project focuses on predicting the next word given a sequence of words, using an RNN-based neural network that learns from text data in a QA format.

The system learns word dependencies and context through training and generates predictions for incomplete sentences, simulating a simple text auto-completion task.

## 🧩 Dataset

Name: Custom QA Dataset

Format: CSV

Content: 100 unique Question-Answer pairs

Columns:

Question

Answer

Example:

Question	Answer
What is AI?	AI stands for Artificial Intelligence.
How are you?	I am fine.

The text from both questions and answers was preprocessed, tokenized, and used to train the next-word prediction model.

## 🧠 Model Architecture

The model uses a Recurrent Neural Network (RNN) architecture with the following components:

Embedding Layer – Converts words into dense vector representations.

RNN Layer – Captures sequential dependencies between words.

Fully Connected Layer – Predicts the next word’s probability distribution.

Activation Function: ReLU or Tanh

Loss Function: CrossEntropyLoss

Optimizer: Adam

## ⚙️ Tech Stack

Language: Python

Framework: PyTorch

# Libraries:

pandas

numpy

torch

torchtext (optional)

sklearn (for preprocessing)

## 🧾 Project Workflow

Data Preprocessing

Loaded QA dataset from CSV

Tokenized and cleaned text

Built vocabulary and word-to-index mapping

Model Building

Implemented a custom RNN model using PyTorch

Defined forward propagation and loss computation

Training

Trained the model on tokenized sequences

Tuned hyperparameters (epochs, learning rate, hidden size)

Prediction

Given an input phrase, the model predicts the next likely word

# Example:

Input: "What is"
Output: "AI"

# 📈 Results

Successfully predicts the next word for short input sequences.

Demonstrates language understanding and contextual word prediction.

Can be extended to LSTM or GRU for improved accuracy.

## 🧪 How to Run
# 1️⃣ Clone Repository
git clone https://github.com/yourusername/next-word-predictor.git
cd next-word-predictor

# 2️⃣ Install Dependencies
pip install -r requirements.txt

# Run the Model
python main.py

#  Try Predictions

Modify the input_text variable in main.py to test your own sentences.

## 📚 Future Improvements

Replace RNN with LSTM or GRU for better long-term dependency handling

Train on a larger text corpus for improved generalization

Add a web interface using Streamlit for user interaction

Incorporate Beam Search for more accurate predictions

## 🧑‍💻 Author

Saad Hussain
📘 Computer Science Student | Machine Learning Enthusiast
🏁 License

This project is open-source and available under the MIT License
.