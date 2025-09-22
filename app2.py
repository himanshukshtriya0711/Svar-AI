from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load models
rf_model = joblib.load("randomForest_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # same vectorizer

class InputText(BaseModel):
    text: str

def clean_text(text: str) -> str:
    return text.lower().strip()

@app.post("/predict_rf")
def predict_rf(input: InputText):
    text = input.text
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    
    # Prediction
    pred = rf_model.predict(vec)[0]
    prob = max(rf_model.predict_proba(vec)[0])
    
    return {"label": pred, "confidence": round(prob, 2)}

@app.get("/")
def root():
    return {"message": "Welcome to SvaraAI API!"}
