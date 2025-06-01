from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import string

# Text preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB(alpha=0.5)  # Added smoothing parameter
    model.fit(X_train, y_train)
    # Evaluate model with additional metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    return model, X_test, y_test

# Predict with probability threshold
def predict(model, vectorizer, new_message, threshold=0.6):
    new_message_processed = preprocess_text(new_message)
    new_message_vectorized = vectorizer.transform([new_message_processed])
    proba = model.predict_proba(new_message_vectorized)[0]
    spam_index = list(model.classes_).index('spam')
    spam_proba = proba[spam_index]
    if spam_proba >= threshold:
        return 'spam'
    else:
        return 'ham'
