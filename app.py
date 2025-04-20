import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
import asyncio
import uvicorn
from sklearn.linear_model import LinearRegression
import numpy as np
from pydantic import BaseModel, Field

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score


# Tworzenie aplikacji FastAPI
app = FastAPI()

# Read environment variable
api_key = os.environ.get('API_KEY', 'YOUR_DEFAULT_API_KEY')

# Dodanie middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Zezwalaj na połączenia z dowolnego źródła
    allow_credentials=True,
    allow_methods=["*"],  # Zezwalaj na wszystkie metody HTTP
    allow_headers=["*"],  # Zezwalaj na wszystkie nagłówki
)

# Definiowanie modelu żądania do przewidywania
class PredictionRequest(BaseModel):
    feature: float = Field(..., description="Wymagana cecha wejściowa dla modelu")

# Tworzenie i trenowanie modelu regresji liniowej
model = LinearRegression()

# Dane do trenowania (cechy i wartości docelowe)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Trenowanie modelu
model.fit(X, y)

# Definiowanie głównego endpointu
@app.get("/")
async def root():
    return {"hello": "world"}

# Endpoint przewidywania
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Weryfikacja, czy 'feature' została przekazana (Pydantic automatycznie to sprawdza)
        prediction = model.predict([[request.feature]])
        return {"prediction": prediction[0], "api_key": api_key}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Błąd podczas przetwarzania danych: {str(e)}"
        )

# Endpoint sprawdzający stan serwera
@app.get("/health")
async def health():
    return {"status": "ok"}

# Endpoint zwracający informacje o modelu
@app.get("/info")
async def info():
    return {
        "model_type": "Linear Regression",
        "number_of_features": 1,
        "trained_on_samples": len(X)
    }

def train_and_predict():
    """
    Trains a simple model on the breast cancer dataset and returns predictions and accuracy.
    """
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()  # Using LinearRegression for simplicity
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Convert predictions to class labels (0 or 1)
    predicted_labels = np.round(predictions).astype(int)
    predicted_labels[predicted_labels < 0] = 0
    predicted_labels[predicted_labels > 1] = 1
    
    accuracy = accuracy_score(y_test, predicted_labels)
    return predicted_labels, accuracy

def get_accuracy():
    """
    Returns the accuracy of the model.
    """
    _, accuracy = train_and_predict()
    return accuracy

# Uruchomienie serwera za pomocą Uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Using port: {port}")
    print(f"PORT environment variable: {os.environ.get('PORT')}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
