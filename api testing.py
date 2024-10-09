import requests

# URL of the Flask API
url = "http://127.0.0.1:5000/predict"

# Example input data for prediction
input_data = {
    "input": [6, 148, 72, 35, 0, 33.6, 0.627, 50]  # Replace with actual input data
}

# Send POST request
response = requests.post(url, json=input_data)

# Check the status code and print response content
if response.status_code == 200:
    # Try to print the JSON response
    try:
        print(response.json())
    except ValueError:
        print("Response content is not valid JSON:", response.text)
else:
    # Print the status code and the response text for debugging
    print(f"Failed with status code {response.status_code}: {response.text}")
