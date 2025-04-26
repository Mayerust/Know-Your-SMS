import streamlit as st
import pickle
from scripts.preprocess import transform_text

# Display only the project name
st.title("Know Your SMS")

input_sms = st.text_input("Enter the SMS")

# Load pre-trained artifacts from the models directory
vectorizer = pickle.load(open("models/vectorizer.pkl", 'rb'))
model = pickle.load(open("models/model.pkl", 'rb'))

if st.button('Predict'):
    # Preprocess and vectorize the input message
    processed_text = transform_text(input_sms)
    vector_input = vectorizer.transform([processed_text])
    
    # Predict and display the result
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")