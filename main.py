import pandas as pd
from data_handler import load_data, preprocess_data
from model import train_model, predict
from feedback_logger import log_feedback
from sklearn.metrics import accuracy_score  # Import accuracy_score
import tkinter as tk

def display_output(message, prediction):
    # Create a new window to display the output
    output_window = tk.Tk()
    output_window.title("Spam Detection Result")
    
    # Create a label to show the message and prediction
    result_label = tk.Label(output_window, text=f'Message: "{message}"\nClassified as: {prediction}', padx=20, pady=20)
    result_label.pack()

    # Create a button to close the window
    close_button = tk.Button(output_window, text="Close", command=output_window.destroy)
    close_button.pack(pady=10)

    output_window.mainloop()

def main():
    print("Welcome to the SMS Spam Detector!")
    print("This program will classify SMS messages as spam or not spam.")
    
    # Load and preprocess data
    data = load_data('spam_data.csv')
    X, y, vectorizer = preprocess_data(data)

    # Train the model
    model, X_test, y_test = train_model(X, y)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    print(f'Accuracy: {accuracy}')

    # Allow user to classify multiple messages
    while True:
        random_message = data.sample(n=1)['message'].values[0]
        prediction = predict(model, vectorizer, random_message)

        # Display the output in a separate window
        display_output(random_message, prediction)

        # Check if the message is potentially legitimate
        if "order" in random_message.lower() or "delivery" in random_message.lower():
            user_input = input("This message seems legitimate. Should it be classified as spam? (yes/no): ").strip().lower()
            log_feedback(random_message, prediction, user_input)
        else:
            print("No user input needed for this message.")

        continue_input = input("Do you want to classify another message? (yes/no): ").strip().lower()
        if continue_input != 'yes':
            break

    print("Thank you for using the SMS Spam Detector!")

if __name__ == "__main__":
    main()
