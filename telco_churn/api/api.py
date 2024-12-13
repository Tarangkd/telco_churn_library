''' Final Project '''
import sys
import os
sys.path.append('c:/Users/Enzo/Documents/BSE/T1/COMPUTING_DS/Final_Project/Final-Project/telco-churn-library')

import joblib
import pandas as pd
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from telco_churn.data_preprocessing.preprocessor import HandleMissingValues, NormalizeData, EncodeCategoricalData, HandleOutliers, PreprocessingPipeline
from telco_churn.feature_engineering.feature_engineering import StatisticalFeatures, CategoricalEncoding, InteractionFeatures, TemporalFeatures, DerivedFeatures, FeaturePipeline

# Create FastAPI application
app = FastAPI()

# Load the final model
model = joblib.load('telco_churn/pipeline/trained_model.pkl')

def preprocess_data(input_df):
    # Apply preprocessing transformations and feature engineering pipeline
    preprocessing_pipeline = PreprocessingPipeline(preprocessors=[
        HandleMissingValues(strategy='mean'),
        NormalizeData(method='minmax'),
        EncodeCategoricalData(),
        HandleOutliers(method='iqr')
    ])
    
    # Preprocess the input data
    processed_data = preprocessing_pipeline.apply(input_df)

    # Apply feature engineering
    feature_pipeline = FeaturePipeline(transformers=[
        StatisticalFeatures(group_by_column=None),
        CategoricalEncoding(),
        InteractionFeatures(),
        TemporalFeatures(),
        DerivedFeatures()
    ])
    
    # Apply additional features
    final_data = feature_pipeline.apply(processed_data)
    
    return final_data

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Validate that the file is of JSON type
    if file.content_type != 'application/json':
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .json file.")
    
    # Read and process the JSON file
    try:
        contents = await file.read()
        input_data = json.loads(contents)  # Convert the content into a dictionary
        
        # Convert the JSON into a DataFrame
        input_df = pd.DataFrame(input_data)  # Convert the dictionary into a DataFrame
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading the JSON file: {e}")
    
    # Apply transformations just like during training
    try:
        transformed_data = preprocess_data(input_df)

        # Perform the prediction
        prediction = model.predict(transformed_data)
        probability = model.predict_proba(transformed_data) if hasattr(model, "predict_proba") else None

        response = {
            "prediction": int(prediction[0]),  # Convert the prediction to int for JSON
            "probability": probability[0].tolist() if probability is not None else "Not available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
    
    return response

