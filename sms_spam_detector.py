import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import tkinter as tk
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download NLTK resources if not already downloaded
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords and apply stemming
    words = text.split()
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    return text

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Apply preprocessing to messages
    data['message'] = data['message'].apply(preprocess_text)
    return data

# Preprocess data
def preprocess_data(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['message'])
    y = data['label']
    return X, y, vectorizer

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

# Secondary heuristic to detect suspicious spam patterns
def is_suspicious_spam(text):
    # Check for excessive numbers, special characters, or common spam phrases
    if len(re.findall(r'\d', text)) > 5:
        return True
    if len(re.findall(r'[!@#$%^&*()_+=\-{}\[\]:;"\'<>.,?\\/|~`]', text)) > 5:
        return True
    spam_phrases = ['free', 'win', 'winner', 'cash', 'prize', 'urgent', 'congratulations', 'claim', 'offer',
                    'account', 'credited', 'credit', 'lakhs', 'credited with', 'your account', 'amount']
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in spam_phrases):
        return True
    return False

# Predict with probability threshold and secondary heuristic
def predict(model, vectorizer, new_message, threshold=0.6):
    new_message_processed = preprocess_text(new_message)
    new_message_vectorized = vectorizer.transform([new_message_processed])
    proba = model.predict_proba(new_message_vectorized)[0]
    spam_index = list(model.classes_).index('spam')
    spam_proba = proba[spam_index]
    if spam_proba >= threshold or is_suspicious_spam(new_message):
        return 'spam', spam_proba
    else:
        return 'ham', spam_proba

# Log user feedback
def log_feedback(message, prediction, user_input):
    feedback_data = {
        "message": message,
        "prediction": prediction,
        "user_input": user_input
    }
    with open('user_feedback.json', 'a') as f:
        f.write(json.dumps(feedback_data) + '\n')

def classify_message():
    user_message = message_entry.get()
    if not user_message:
        result_label.config(text="Please enter a message to classify.", fg="red")
        return

    prediction, confidence = predict(model, vectorizer, user_message)
    color = "green" if prediction == "ham" else "red"
    result_label.config(text=f'Message: "{user_message}"\nClassified as: {prediction} (Confidence: {confidence:.2f})', fg=color)

    # Log feedback if user selected an option
    feedback = feedback_var.get()
    if feedback in ("yes", "no"):
        log_feedback(user_message, prediction, feedback)

def main():
    global model, vectorizer, message_entry, result_label, feedback_var

    # Load and preprocess data
    data = load_data('spam_data.csv')
    X, y, vectorizer = preprocess_data(data)

    # Train the model
    model, X_test, y_test = train_model(X, y)

    # Create the main window
    window = tk.Tk()
    window.title("SMS Spam Detector")
    window.geometry("500x300")
    window.configure(bg="#f0f0f0")

    # Title label
    title_label = tk.Label(window, text="SMS Spam Detector", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
    title_label.pack(pady=(20, 10))

    # Create input field
    message_entry = tk.Entry(window, width=50, font=("Helvetica", 12))
    message_entry.pack(pady=10, padx=20)

    # Create classify button
    classify_button = tk.Button(window, text="Classify Message", command=classify_message,
                                font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", activebackground="#45a049",
                                relief=tk.FLAT, padx=10, pady=5)
    classify_button.pack(pady=10)

    # Frame for result label with border and padding
    result_frame = tk.Frame(window, bg="white", bd=2, relief=tk.SOLID, padx=10, pady=10)
    result_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

    # Create label for displaying results
    result_label = tk.Label(result_frame, text="", font=("Helvetica", 12), bg="white", justify=tk.LEFT)
    result_label.pack()

    # Create feedback options
    feedback_var = tk.StringVar(value="no")
    feedback_frame = tk.Frame(window, bg="#f0f0f0")
    feedback_frame.pack(pady=10)
    tk.Label(feedback_frame, text="Was the classification correct?", bg="#f0f0f0", font=("Helvetica", 10)).pack(side=tk.LEFT)
    tk.Radiobutton(feedback_frame, text="Yes", variable=feedback_var, value="yes", bg="#f0f0f0").pack(side=tk.LEFT)
    tk.Radiobutton(feedback_frame, text="No", variable=feedback_var, value="no", bg="#f0f0f0").pack(side=tk.LEFT)

    # Start the Tkinter event loop
    window.mainloop()

if __name__ == "__main__":
    main()
