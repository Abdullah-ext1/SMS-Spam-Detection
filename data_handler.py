import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess data
def preprocess_data(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['message'])
    y = data['label']
    return X, y, vectorizer
