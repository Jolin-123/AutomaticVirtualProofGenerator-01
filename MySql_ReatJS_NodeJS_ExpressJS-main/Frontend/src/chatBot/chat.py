from flask import Flask, request, jsonify
import random
import torch
import json
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)

# Load the model (same as before)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE, weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

# Chatbot response function
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)  # Move to the appropriate device

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand... "

# API endpoint to receive message and return response
@app.route('/get_response', methods=['POST'])
def api_get_response():
    data = request.get_json()
    user_message = data.get("message")
    if user_message:
        response = get_response(user_message)
        return jsonify({'response': response})
    return jsonify({'response': "Error: No message provided"}), 400

if __name__ == "__main__":
    app.run(port=5000)
