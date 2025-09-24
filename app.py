from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from typing import Optional, Tuple, Dict, Any
import re
from sklearn.exceptions import NotFittedError

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

def _ensure_legacy_loaded() -> None:
    """Load legacy vectorizer/models if not already loaded."""
    global vectorizer, svm_model, rf_model
    if vectorizer is None:
        vectorizer = _safe_load("tfidf_vectorizer.pkl")
    if svm_model is None:
        svm_model = _safe_load("svm_model.pkl")
    if rf_model is None:
        rf_model = _safe_load("randomForest_model.pkl")

# Preferred: single-file pipelines
svm_pipeline = _safe_load("svm_best_pipeline.pkl")
rf_pipeline = _safe_load("rf_best_pipeline.pkl")

# Backward compatibility: also load separate artifacts if available
_ensure_legacy_loaded()

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

# Helpers to predict safely with a pipeline or legacy model
def _predict_with_pipeline(pipeline, cleaned_text: str) -> Tuple[str, float]:
    """Return (label, confidence) using a fitted sklearn Pipeline.
    Raises NotFittedError if pipeline isn't fitted yet.
    """
    # Predict label
    pred = pipeline.predict([cleaned_text])[0]
    # Confidence via predict_proba if available; else try decision_function; else 1.0
    try:
        proba = pipeline.predict_proba([cleaned_text])[0]
        conf = float(max(proba))
    except Exception:
        try:
            import numpy as np
            dec = pipeline.decision_function([cleaned_text])
            m = float(np.max(dec)) if hasattr(dec, "__len__") else float(dec)
            conf = float(1 / (1 + np.exp(-m)))
        except Exception:
            conf = 1.0
    return pred, conf

def _predict_with_legacy(model, vec, cleaned_text: str) -> Tuple[str, float]:
    """Return (label, confidence) using legacy vectorizer + model."""
    X = vec.transform([cleaned_text])
    pred = model.predict(X)[0]
    try:
        conf = float(max(model.predict_proba(X)[0]))
    except Exception:
        conf = 1.0
    return pred, conf

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
        try:
            pred, prob = _predict_with_pipeline(svm_pipeline, cleaned)
            return {"model": "SVM (pipeline)", "label": pred, "confidence": round(prob, 2)}
        except NotFittedError:
            # Fall through to legacy path if available
            _ensure_legacy_loaded()
    # Legacy path
    if vectorizer is None or svm_model is None:
        raise HTTPException(status_code=503, detail="SVM pipeline/model not available")
    pred, prob = _predict_with_legacy(svm_model, vectorizer, cleaned)
    return {"model": "SVM", "label": pred, "confidence": round(prob, 2)}

# RandomForest prediction
@app.post("/predict_rf")
def predict_rf(input: InputText):
    text = input.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    cleaned = clean_text(text)
    if rf_pipeline is not None:
        try:
            pred, prob = _predict_with_pipeline(rf_pipeline, cleaned)
            return {"model": "RandomForest (pipeline)", "label": pred, "confidence": round(prob, 2)}
        except NotFittedError:
            _ensure_legacy_loaded()
    # Legacy path
    if vectorizer is None or rf_model is None:
        raise HTTPException(status_code=503, detail="RF pipeline/model not available")
    pred, prob = _predict_with_legacy(rf_model, vectorizer, cleaned)
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
        try:
            svm_pred, svm_prob = _predict_with_pipeline(svm_pipeline, cleaned)
            result["svm"] = {"label": svm_pred, "confidence": round(svm_prob, 2)}
        except NotFittedError:
            _ensure_legacy_loaded()
            if vectorizer is not None and svm_model is not None:
                svm_pred, svm_prob = _predict_with_legacy(svm_model, vectorizer, cleaned)
                result["svm"] = {"label": svm_pred, "confidence": round(svm_prob, 2)}
    elif vectorizer is not None and svm_model is not None:
        svm_pred, svm_prob = _predict_with_legacy(svm_model, vectorizer, cleaned)
        result["svm"] = {"label": svm_pred, "confidence": round(svm_prob, 2)}

    # RF branch
    if rf_pipeline is not None:
        try:
            rf_pred, rf_prob = _predict_with_pipeline(rf_pipeline, cleaned)
            result["random_forest"] = {"label": rf_pred, "confidence": round(rf_prob, 2)}
        except NotFittedError:
            _ensure_legacy_loaded()
            if vectorizer is not None and rf_model is not None:
                rf_pred, rf_prob = _predict_with_legacy(rf_model, vectorizer, cleaned)
                result["random_forest"] = {"label": rf_pred, "confidence": round(rf_prob, 2)}
    elif vectorizer is not None and rf_model is not None:
        rf_pred, rf_prob = _predict_with_legacy(rf_model, vectorizer, cleaned)
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
        try:
            pred, proba = _predict_with_pipeline(svm_pipeline, cleaned)
            return {"model": "SVM (pipeline)", "label": pred, "confidence": round(proba, 2)}
        except NotFittedError:
            _ensure_legacy_loaded()
    # fallback to RF pipeline
    if rf_pipeline is not None:
        try:
            pred, proba = _predict_with_pipeline(rf_pipeline, cleaned)
            return {"model": "RandomForest (pipeline)", "label": pred, "confidence": round(proba, 2)}
        except NotFittedError:
            _ensure_legacy_loaded()
    # legacy fallback
    if vectorizer is not None and svm_model is not None:
        pred, proba = _predict_with_legacy(svm_model, vectorizer, cleaned)
        return {"model": "SVM", "label": pred, "confidence": round(proba, 2)}
    if vectorizer is not None and rf_model is not None:
        pred, proba = _predict_with_legacy(rf_model, vectorizer, cleaned)
        return {"model": "RandomForest", "label": pred, "confidence": round(proba, 2)}
    raise HTTPException(status_code=503, detail="No models available for prediction")
