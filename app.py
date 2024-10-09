import torch
import pandas as pd
from flask import Flask, request, jsonify
import joblib
import torch.nn as nn


# Define the neural network model
class DiabetesModel(nn.Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# Create Flask app
app = Flask(__name__)

# Load the saved model and scaler
model = DiabetesModel()
model.load_state_dict(torch.load('diabetes_model.pth', weights_only=True))
model.eval()
scaler = joblib.load('scaler_diabetes.pkl')


@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_features = data['features']

    # Standardize input data
    input_data = scaler.transform([input_features])
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()

    return jsonify({'prediction': prediction})


@app.route('/evaluate', methods=['GET'])
def evaluate():
    # Load the dataset for evaluation
    data = pd.read_csv('diabetes.csv')  # Ensure this file is available
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Standardize the data
    X_scaled = scaler.transform(X)

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_tensor)
        test_predictions = (test_outputs >= 0.5).float()  # Apply threshold
        accuracy = (test_predictions.eq(y_tensor).sum().item()) / y_tensor.size(0)

    return jsonify({'test_accuracy': accuracy})


if __name__ == '__main__':
    app.run(debug=True)
