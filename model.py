from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Predict
def predict(model, vectorizer, new_message):
    new_message_vectorized = vectorizer.transform([new_message])
    prediction = model.predict(new_message_vectorized)
    return prediction[0]
