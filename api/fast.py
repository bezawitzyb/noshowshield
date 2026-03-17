from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from eda_package.model import train_model
from eda_package.preprocessor import transform_preprocessor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and preprocessor at startup
model, preprocessor = train_model()

# Define request body schema
class PredictionRequest(BaseModel):
    data: dict

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

# Endpoint for POST https://your-domain.com/predict
# POST: {"data": {feature1: value1, feature2: value2, ..., feature40: value40}}
@app.post("/predict")
def predict_post(request: PredictionRequest):
    """POST endpoint - pass all 40 features as JSON body"""
    return _predict(request.data)

def _predict(data: dict):
    """Shared prediction logic"""
    try:
        # Convert input dictionary to DataFrame
        X = pd.DataFrame([data])

        # Transform using the preprocessor
        X_processed = transform_preprocessor(X, preprocessor)

        # Get prediction
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0]

        return {
            'prediction': int(prediction),
            'probability': {
                'no_cancel': float(probability[0]),
                'cancel': float(probability[1])
            },
            'inputs': data
        }
    except Exception as e:
        return {
            'error': str(e),
            'inputs': data
        }
