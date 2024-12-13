import sys
import os
sys.path.append('c:/Users/Enzo/Documents/BSE/T1/COMPUTING_DS/Final_Project/Final-Project/telco-churn-library')

import requests
import json

# Set the API endpoint URL
url = "http://127.0.0.1:8000/predict/"

# Send the POST request
try:
    # Open the input JSON file
    with open("telco_churn/api/json_file.json", "r") as file:
        # Send the POST request with the JSON file
        response = requests.post(url, files={"file": ("json_file.json", file, "application/json")})
    
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


