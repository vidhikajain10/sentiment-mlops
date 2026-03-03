import mlflow.sklearn
from fastapi import FastAPI
import pandas as pd

app = FastAPI()

model = mlflow.sklearn.load_model("models:/Sentiment_MLOps/1")

@app.post("/predict")
def predict(text: str):
    prediction = model.predict([text])
    return {"sentiment": prediction[0]}
