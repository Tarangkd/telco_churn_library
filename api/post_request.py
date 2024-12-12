import requests
import json

# Set the API endpoint URL
url = "http://127.0.0.1:8000/predict"

# Input data for a single prediction
input_data = {
    "gender": "Male",              
    "tenure": 12,                  
    "MonthlyCharges": 70.5,        
    "TotalCharges": 845.5          
}

try:
    # Convert input data to JSON format
    json_data = json.dumps(input_data)
    
    # Send the POST request with the JSON data
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json_data, headers=headers)
    
    # Check for HTTP errors
    response.raise_for_status()
    
    # Parse and print the response
    result = response.json()
    
    # Using .get() to avoid KeyError if any field is missing
    prediction = result.get("prediction", "Prediction not available")
    probability = result.get("probability", "Probability not available")
    
    print("Prediction:", prediction)
    print("Probability:", probability)

except requests.exceptions.RequestException as e:
    print(f"API request error: {e}")
except json.JSONDecodeError:
    print("Error decoding the response from the API.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
