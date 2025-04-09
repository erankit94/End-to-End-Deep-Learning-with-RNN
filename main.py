import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
rev_dict = {value:key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([rev_dict.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## prediction func
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

## streamlit
import streamlit as st
st.title("IMDB Movie review sentiment analysis")
st.write("enter a movie review")

# user input
user_input = st.text_area("movie review")

if st.button("classify"):
    preprocessed_input = preprocess_text(user_input)
    # make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    # display results:
    st.write(f'Sentiment: {sentiment}')
    st.write(f'prediction_score: {prediction[0][0]}')
else:
    st.write('please enter movie review')
