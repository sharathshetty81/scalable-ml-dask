from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pipeline.config import MODEL_PATH

app = FastAPI()
model = joblib.load(MODEL_PATH)

# Define expected input format
class Features(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.get("/")
def root():
    return {"message": "✅ Dask ML API is running. Use POST /predict"}

@app.post("/predict")
def predict(data: Features):
    features = np.array([[data.feature1, data.feature2, data.feature3]])
    prediction = int(model.predict(features)[0])
    return {"prediction": prediction}

