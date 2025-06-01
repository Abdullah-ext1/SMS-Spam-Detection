import pandas as pd
from data_handler import load_data, preprocess_data
from model_updated import train_model, predict
from feedback_logger import log_feedback
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

def classify_message():
    user_message = message_entry.get()
    if not user_message:
        messagebox.showerror("Input Error", "Please enter a message to classify.")
        return

    prediction = predict(model, vectorizer, user_message, threshold=0.8)
    # Color the result text based on prediction
    color = "green" if prediction.lower() == "ham" else "red"
    result_label.config(text=f'Message: "{user_message}"\nClassified as: {prediction}', fg=color)

    # Log feedback
    if "order" in user_message.lower() or "delivery" in user_message.lower():
        feedback = feedback_var.get()
        log_feedback(user_message, prediction, feedback)

def main():
    global model, vectorizer, message_entry, result_label, feedback_var

    # Load and preprocess data
    data = load_data('spam_data.csv')
    X, y, vectorizer = preprocess_data(data)

    # Train the model
    model, X_test, y_test = train_model(X, y)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

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
