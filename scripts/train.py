import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from scripts.preprocess import transform_text  # Adjust based on your structure

# Load and prepare the dataset
data = pd.read_csv('data/spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Preprocess each message
data['processed'] = data['message'].apply(transform_text)

# Separate features and target
X = data['processed']
y = data['label']

# Split data for training and evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using the bag-of-words approach
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate the model
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# Save the trained vectorizer and model
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
pickle.dump(model, open('models/model.pkl', 'wb'))