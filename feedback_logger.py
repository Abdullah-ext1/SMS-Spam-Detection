import json

# Log user feedback
def log_feedback(message, prediction, user_input):
    feedback_data = {
        "message": message,
        "prediction": prediction,
        "user_input": user_input
    }
    with open('user_feedback.json', 'a') as f:
        f.write(json.dumps(feedback_data) + '\n')
