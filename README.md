# Know Your SMS

## Overview
Know Your SMS is a machine learning project that classifies an SMS message as spam or not. Built in Python using classical NLP techniques, it provides a streamlined web interface via Streamlit.

## Technologies
- Python
- Scikit-learn
- Pandas
- Numpy
- Streamlit
- NLTK

## Features
- Data collection and cleaning
- Text preprocessing: tokenization, stopword removal, and stemming
- Exploratory Data Analysis for insights into messaging data
- Model training using a Naive Bayes classifier
- Deployment as a web application with Streamlit

### Data Collection
The underlying dataset is sourced from the SMS Spam Collection on Kaggle, featuring over 5,500 labeled messages. You can access it on [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

### Text Preprocessing
Messages are converted to lowercase, tokenized, filtered to remove stopwords and punctuation, and then stemmed.

### Model Training
The training script evaluates multiple metrics, and the final model (based on Naive Bayes) is saved along with its vectorizer.

### Web Application
The web app lets users input an SMS message and instantly get a prediction on whether it is spam or not.

## Getting Started

### Clone the Repository
```bash
git clone <repository_url>
