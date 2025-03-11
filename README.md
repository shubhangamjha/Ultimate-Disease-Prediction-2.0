# Ultimate-Disease-Prediction-2.0

# Diabetes Prediction API

A Flask-based REST API for predicting diabetes risk using a PyTorch neural network model.

## Features
- Predict diabetes risk based on 8 health parameters
- Evaluate model accuracy on test dataset
- Pre-trained PyTorch model integration
- Data preprocessing with standardization
- Simple JSON-based interface

## Prerequisites
- Python 3.6+
- pip package manager

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-prediction-api.git
cd diabetes-prediction-api

pip install -r requirements.txt

curl -X POST -H "Content-Type: application/json" \
-d '{"features": [1, 85, 66, 29, 0, 26.6, 0.351, 31]}' \
http://localhost:5000/predict

curl http://localhost:5000/evaluate


**Important Notes:**
1. Make sure you have these files in your project directory:
   - `diabetes_model.pth` (pretrained model)
   - `scaler_diabetes.pkl` (preprocessing scaler)
   - `diabetes.csv` (dataset)

2. For production use:
   - Turn off debug mode (`debug=False`)
   - Use proper WSGI server (Gunicorn/uWSGI)
   - Add authentication/rate limiting
   - Do not use the evaluation endpoint in production (uses entire dataset)

You can customize this template further based on your specific implementation details and requirements.
