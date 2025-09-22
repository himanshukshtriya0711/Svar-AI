from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI(title="SvaraAI Multi-Model API")

# Load models
try:
    svm_model = joblib.load("svm_model.pkl")
    rf_model = joblib.load("randomForest_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load models/vectorizer: {e}")

# Input schema
class InputText(BaseModel):
    text: str

# Basic text cleaning
def clean_text(text: str) -> str:
    return text.lower().strip()

# Root route
@app.get("/")
def root():
    return {"message": "Welcome to SvaraAI Multi-Model API!"}

# SVM prediction
@app.post("/predict_svm")
def predict_svm(input: InputText):
    text = input.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    vec = vectorizer.transform([clean_text(text)])
    pred = svm_model.predict(vec)[0]
    prob = max(svm_model.predict_proba(vec)[0])
    return {"model": "SVM", "label": pred, "confidence": round(prob, 2)}

# RandomForest prediction
@app.post("/predict_rf")
def predict_rf(input: InputText):
    text = input.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    vec = vectorizer.transform([clean_text(text)])
    pred = rf_model.predict(vec)[0]
    prob = max(rf_model.predict_proba(vec)[0])
    return {"model": "RandomForest", "label": pred, "confidence": round(prob, 2)}

# Combined prediction
@app.post("/predict_all")
def predict_all(input: InputText):
    text = input.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    vec = vectorizer.transform([clean_text(text)])

    svm_pred = svm_model.predict(vec)[0]
    svm_prob = max(svm_model.predict_proba(vec)[0])

    rf_pred = rf_model.predict(vec)[0]
    rf_prob = max(rf_model.predict_proba(vec)[0])

    return {
        "svm": {"label": svm_pred, "confidence": round(svm_prob, 2)},
        "random_forest": {"label": rf_pred, "confidence": round(rf_prob, 2)}
    }
