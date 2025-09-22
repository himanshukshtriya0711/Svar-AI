# SvaraAI - Reply Classification System

A machine learning-powered text classification system that categorizes customer replies into POSITIVE, NEGATIVE, or NEUTRAL sentiment using Support Vector Machine (SVM) and Random Forest models.

## 📋 Project Overview

This project implements a complete machine learning pipeline for text classification:

- **Dataset**: 2,129 customer reply samples with sentiment labels
- **Models**: SVM (Linear kernel) and Random Forest (200 estimators)
- **Text Processing**: TF-IDF vectorization with stopword removal
- **API**: FastAPI-based REST endpoints for real-time predictions
- **Features**: Single model predictions and ensemble predictions

## 🏗️ Project Structure

```
svarai/
├── SVM.ipynb                          # Jupyter notebook with ML pipeline
├── reply_classification_dataset.csv   # Training dataset (2,129 samples)
├── app.py                             # Multi-model FastAPI application
├── app2.py                            # Random Forest only FastAPI application
├── svm_model.pkl                      # Trained SVM model (121 KB)
├── randomForest_model.pkl             # Trained Random Forest model (7.5 MB)
├── tfidf_vectorizer.pkl               # TF-IDF vectorizer (6.6 KB)
└── __pycache__/                       # Python cache files
```

## 🔧 Tech Stack

- **Machine Learning**: scikit-learn, NLTK
- **API Framework**: FastAPI, Uvicorn
- **Data Processing**: pandas, numpy
- **Text Processing**: TF-IDF vectorization, regex, stopwords removal
- **Model Persistence**: joblib

## 📊 Model Performance

### Support Vector Machine (Linear)
- **Features**: TF-IDF vectors (max 5,000 features)
- **Kernel**: Linear
- **Size**: 121 KB

### Random Forest
- **Estimators**: 200 trees
- **Features**: TF-IDF vectors (max 5,000 features)
- **Size**: 7.5 MB

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### 1. Clone/Download the Project
```bash
# Navigate to your project directory
cd c:\Users\himan\OneDrive\Desktop\svarai
```

### 2. Install Required Dependencies
```bash
pip install fastapi uvicorn scikit-learn pandas nltk joblib pydantic
```

### 3. Download NLTK Data (Required for text preprocessing)
```python
import nltk
nltk.download('stopwords')
```

### 4. Verify Model Files
Ensure these files are present:
- `svm_model.pkl`
- `randomForest_model.pkl`  
- `tfidf_vectorizer.pkl`

## 🔥 Running the API Locally

### Option 1: Multi-Model API (Recommended)
```bash
# Run the comprehensive API with both models
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Random Forest Only API
```bash
# Run the Random Forest only API
uvicorn app2:app --reload --host 0.0.0.0 --port 8001
```

### Access the API
- **API URL**: http://localhost:8000 (or 8001 for app2)
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 📡 API Endpoints

### Multi-Model API (`app.py`)

#### 1. Root Endpoint
```http
GET /
```
**Response:**
```json
{
  "message": "Welcome to SvaraAI Multi-Model API!"
}
```

#### 2. SVM Prediction
```http
POST /predict_svm
```
**Request Body:**
```json
{
  "text": "This looks promising, let's schedule a meeting!"
}
```
**Response:**
```json
{
  "model": "SVM",
  "label": "POSITIVE",
  "confidence": 0.89
}
```

#### 3. Random Forest Prediction
```http
POST /predict_rf
```
**Request Body:**
```json
{
  "text": "Please remove me from your mailing list"
}
```
**Response:**
```json
{
  "model": "RandomForest",
  "label": "NEGATIVE",
  "confidence": 0.92
}
```

#### 4. Combined Predictions (Ensemble)
```http
POST /predict_all
```
**Request Body:**
```json
{
  "text": "Could you clarify the pricing details?"
}
```
**Response:**
```json
{
  "svm": {
    "label": "NEUTRAL",
    "confidence": 0.76
  },
  "random_forest": {
    "label": "NEUTRAL", 
    "confidence": 0.81
  }
}
```

### Random Forest API (`app2.py`)

#### Random Forest Prediction
```http
POST /predict_rf
```
**Request Body:**
```json
{
  "text": "Excited to explore this further!"
}
```
**Response:**
```json
{
  "label": "POSITIVE",
  "confidence": 0.94
}
```

## 💡 Usage Examples

### Using cURL
```bash
# Test SVM prediction
curl -X POST "http://localhost:8000/predict_svm" \
     -H "Content-Type: application/json" \
     -d '{"text": "This solution looks interesting, send me more details"}'

# Test ensemble prediction
curl -X POST "http://localhost:8000/predict_all" \
     -H "Content-Type: application/json" \
     -d '{"text": "Not interested in this service"}'
```

### Using Python requests
```python
import requests

# API endpoint
url = "http://localhost:8000/predict_all"

# Test data
data = {"text": "Let's schedule a demo to discuss further"}

# Make prediction
response = requests.post(url, json=data)
result = response.json()

print(f"SVM: {result['svm']['label']} (confidence: {result['svm']['confidence']})")
print(f"RF: {result['random_forest']['label']} (confidence: {result['random_forest']['confidence']})")
```

## 📈 Model Training Process

The complete machine learning pipeline is documented in `SVM.ipynb`:

1. **Data Loading**: Load 2,129 customer reply samples
2. **Text Preprocessing**: 
   - Convert to lowercase
   - Remove URLs, punctuation, numbers
   - Remove English stopwords
3. **Feature Engineering**: TF-IDF vectorization (max 5,000 features)
4. **Model Training**: Train SVM and Random Forest classifiers
5. **Model Evaluation**: Calculate accuracy and F1 scores
6. **Model Persistence**: Save models using joblib

### Label Distribution
- **POSITIVE**: Customer shows interest/enthusiasm
- **NEGATIVE**: Customer rejection/dissatisfaction  
- **NEUTRAL**: Customer inquiry/neutral response

## 🛠️ Customization

### Modifying Text Preprocessing
Edit the `clean_text()` function in `app.py`:
```python
def clean_text(text: str) -> str:
    # Add your custom preprocessing here
    text = text.lower().strip()
    # Remove special characters, normalize text, etc.
    return text
```

### Adding New Models
1. Train your model in the Jupyter notebook
2. Save using `joblib.dump(model, "your_model.pkl")`
3. Load in FastAPI: `your_model = joblib.load("your_model.pkl")`
4. Create new endpoint following existing patterns

## 🔍 Troubleshooting

### Common Issues

1. **Models not found**
   ```
   RuntimeError: Failed to load models/vectorizer
   ```
   **Solution**: Ensure all `.pkl` files are in the project directory

2. **NLTK data missing**
   ```
   LookupError: Resource stopwords not found
   ```
   **Solution**: Run `import nltk; nltk.download('stopwords')`

3. **Port already in use**
   ```
   OSError: [WinError 10048] Only one usage of each socket address
   ```
   **Solution**: Use a different port: `--port 8001`

### Performance Tips
- Use `--workers 4` for production deployment
- Consider caching predictions for repeated requests
- Monitor memory usage with large Random Forest model

## 📝 Development

### Running in Development Mode
```bash
# Enable auto-reload for development
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### Testing the Models
Use the Jupyter notebook `SVM.ipynb` to:
- Retrain models with new data
- Experiment with hyperparameters
- Evaluate model performance
- Test preprocessing functions

## 🌐 Production Deployment

For production deployment, consider:
- Using Gunicorn with Uvicorn workers
- Setting up reverse proxy (nginx)
- Adding authentication and rate limiting
- Implementing logging and monitoring
- Using environment variables for configuration

