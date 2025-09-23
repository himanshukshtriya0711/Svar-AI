from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from typing import Optional
import re

app = FastAPI(title="SvaraAI Multi-Model API")

# Load pipelines first (preferred), then fall back to legacy artifacts
svm_pipeline: Optional[object] = None
rf_pipeline: Optional[object] = None
svm_model = None
rf_model = None
vectorizer = None

def _safe_load(path: str) -> Optional[object]:
    try:
        return joblib.load(path)
    except Exception:
        return None

# Preferred: single-file pipelines
svm_pipeline = _safe_load("svm_best_pipeline.pkl")
rf_pipeline = _safe_load("rf_best_pipeline.pkl")

# Backward compatibility: separate artifacts
if svm_pipeline is None or rf_pipeline is None:
    vectorizer = _safe_load("tfidf_vectorizer.pkl")
    svm_model = _safe_load("svm_model.pkl")
    rf_model = _safe_load("randomForest_model.pkl")

if (svm_pipeline is None and (svm_model is None or vectorizer is None)) and \
   (rf_pipeline is None and (rf_model is None or vectorizer is None)):
    raise RuntimeError("No valid models found. Expected svm_best_pipeline.pkl/rf_best_pipeline.pkl or legacy tfidf_vectorizer.pkl + svm_model.pkl/randomForest_model.pkl")

# Input schema
class InputText(BaseModel):
    text: str

# Basic text cleaning
def clean_text(text: str) -> str:
    # Mirror training-time basic cleaning; pipeline may also re-clean
    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove urls
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

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
    cleaned = clean_text(text)
    if svm_pipeline is not None:
        pred = svm_pipeline.predict([cleaned])[0]
        # Try predict_proba; fall back to decision_function or 1.0
        prob = None
        try:
            proba = svm_pipeline.predict_proba([cleaned])[0]
            prob = float(max(proba))
        except Exception:
            try:
                # scale decision function to pseudo-probability via sigmoid as a crude proxy
                import numpy as np
                dec = svm_pipeline.decision_function([cleaned])
                # Handle one-vs-rest: take max margin and squash
                m = float(np.max(dec)) if hasattr(dec, "__len__") else float(dec)
                prob = float(1 / (1 + np.exp(-m)))
            except Exception:
                prob = 1.0
        return {"model": "SVM (pipeline)", "label": pred, "confidence": round(prob, 2)}
    # Legacy path
    if vectorizer is None or svm_model is None:
        raise HTTPException(status_code=503, detail="SVM pipeline/model not available")
    vec = vectorizer.transform([cleaned])
    pred = svm_model.predict(vec)[0]
    prob = None
    try:
        prob = float(max(svm_model.predict_proba(vec)[0]))
    except Exception:
        prob = 1.0
    return {"model": "SVM", "label": pred, "confidence": round(prob, 2)}

# RandomForest prediction
@app.post("/predict_rf")
def predict_rf(input: InputText):
    text = input.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    cleaned = clean_text(text)
    if rf_pipeline is not None:
        pred = rf_pipeline.predict([cleaned])[0]
        prob = None
        try:
            prob = float(max(rf_pipeline.predict_proba([cleaned])[0]))
        except Exception:
            prob = 1.0
        return {"model": "RandomForest (pipeline)", "label": pred, "confidence": round(prob, 2)}
    # Legacy path
    if vectorizer is None or rf_model is None:
        raise HTTPException(status_code=503, detail="RF pipeline/model not available")
    vec = vectorizer.transform([cleaned])
    pred = rf_model.predict(vec)[0]
    prob = float(max(rf_model.predict_proba(vec)[0]))
    return {"model": "RandomForest", "label": pred, "confidence": round(prob, 2)}

# Combined prediction
@app.post("/predict_all")
def predict_all(input: InputText):
    text = input.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    cleaned = clean_text(text)
    result = {}
    # SVM branch
    if svm_pipeline is not None:
        svm_pred = svm_pipeline.predict([cleaned])[0]
        try:
            svm_prob = float(max(svm_pipeline.predict_proba([cleaned])[0]))
        except Exception:
            try:
                import numpy as np
                dec = svm_pipeline.decision_function([cleaned])
                m = float(np.max(dec)) if hasattr(dec, "__len__") else float(dec)
                svm_prob = float(1 / (1 + np.exp(-m)))
            except Exception:
                svm_prob = 1.0
        result["svm"] = {"label": svm_pred, "confidence": round(svm_prob, 2)}
    elif vectorizer is not None and svm_model is not None:
        vec = vectorizer.transform([cleaned])
        svm_pred = svm_model.predict(vec)[0]
        try:
            svm_prob = float(max(svm_model.predict_proba(vec)[0]))
        except Exception:
            svm_prob = 1.0
        result["svm"] = {"label": svm_pred, "confidence": round(svm_prob, 2)}

    # RF branch
    if rf_pipeline is not None:
        rf_pred = rf_pipeline.predict([cleaned])[0]
        try:
            rf_prob = float(max(rf_pipeline.predict_proba([cleaned])[0]))
        except Exception:
            rf_prob = 1.0
        result["random_forest"] = {"label": rf_pred, "confidence": round(rf_prob, 2)}
    elif vectorizer is not None and rf_model is not None:
        vec = vectorizer.transform([cleaned])
        rf_pred = rf_model.predict(vec)[0]
        rf_prob = float(max(rf_model.predict_proba(vec)[0]))
        result["random_forest"] = {"label": rf_pred, "confidence": round(rf_prob, 2)}

    if not result:
        raise HTTPException(status_code=503, detail="No models available for prediction")
    return result

# Unified endpoint that prefers tuned SVM pipeline
@app.post("/predict")
def predict(input: InputText):
    text = input.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    cleaned = clean_text(text)
    if svm_pipeline is not None:
        pred = svm_pipeline.predict([cleaned])[0]
        try:
            proba = float(max(svm_pipeline.predict_proba([cleaned])[0]))
        except Exception:
            proba = 1.0
        return {"model": "SVM (pipeline)", "label": pred, "confidence": round(proba, 2)}
    # fallback to RF pipeline
    if rf_pipeline is not None:
        pred = rf_pipeline.predict([cleaned])[0]
        try:
            proba = float(max(rf_pipeline.predict_proba([cleaned])[0]))
        except Exception:
            proba = 1.0
        return {"model": "RandomForest (pipeline)", "label": pred, "confidence": round(proba, 2)}
    # legacy fallback
    if vectorizer is not None and svm_model is not None:
        vec = vectorizer.transform([cleaned])
        pred = svm_model.predict(vec)[0]
        try:
            proba = float(max(svm_model.predict_proba(vec)[0]))
        except Exception:
            proba = 1.0
        return {"model": "SVM", "label": pred, "confidence": round(proba, 2)}
    if vectorizer is not None and rf_model is not None:
        vec = vectorizer.transform([cleaned])
        pred = rf_model.predict(vec)[0]
        proba = float(max(rf_model.predict_proba(vec)[0]))
        return {"model": "RandomForest", "label": pred, "confidence": round(proba, 2)}
    raise HTTPException(status_code=503, detail="No models available for prediction")
