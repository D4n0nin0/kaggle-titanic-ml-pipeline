"""
Titanic Survival Prediction API using FastAPI
----------------------------------------------
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
from typing import Optional

# importar funciones de preprocesamiento
try:
    from src.data_preprocessing import clean_data, encode_data
except ImportError:
    from data_preprocessing import clean_data, encode_data
    
# Crear aplicacion FastAPI
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API para predecir la supervivencia en el Titanic usando un modelo ML entrenado.",
    version="1.0.0"
)

# Definir el modelo de datos de entrada
class PassengerRequest(BaseModel):
    Pclass: int # Clase del pasajero (1, 2, 3)
    Name: str # Nombre del pasajero
    Sex: str # Sexo
    Age: float # Edad
    SibSp: int = 0 # Numero de hermanos/esposas a bordo
    Parch: int = 0 # Numero de padres/hijos a bordo
    Ticket: str # Numero de ticket
    Fare: float # Tarifa del pasajero
    Cabin: Optional[str] = None # Cabina (puede ser nulo)
    Embarked: Optional[str] = None # Puerto de embarque (C, Q, S)
    
class PredictionResponse(BaseModel):
    passenger_id: str
    prediction: int
    probability: float
    survival_status: str
    message: str
    
# Cargar el modelo entrenado y columnas esperadas
def load_model_and_metadata():
    """Cargar el modelo entrenado y las columnas esperadas."""
    
    try:
        models_dir = Path('models')
        
        model_path = models_dir / 'best_model.pkl'
        if not model_path.exists():
            raise FileNotFoundError("Modelo no encontrado. Ejecuta src/train.py primero.")
        
        columns_path = models_dir / 'expected_columns.pkl'
        if not columns_path.exists():
            raise FileNotFoundError("Metadatos del modelo no encontrados")
        
        model = joblib.load(model_path)
        expected_columns = joblib.load(columns_path)
        
        return model, expected_columns
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo: {str(e)}")
    
    
# Endpoint de salud
@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Titanic Survival Prediction API is running.",
        "status": "active",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict",
            "health": "/health"
        }
    }
    
@app.get("/health", tags=["Health"])
async def health_check():
    try:
        model, expected_columns = load_model_and_metadata()
        return {
            "status": "healthy",
            "model_loaded": True,
            "expected_columns_count": len(expected_columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unhealthy: {str(e)} ")
    
# Endpoint principal de prediccion
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_survival(passenger: PassengerRequest):
    """
    Predice si un pasajero del Titanic hubiera sobrevivido o no.
    
    - **Pclass**: Clase del pasajero (1, 2, 3)
    - **Sex**: Sexo (male, female)
    - **Age**: Edad en años
    - **SibSp**: Numero de hermanos/esposas a bordo
    - **Parch**: Numero de padres/hijos a bordo
    - **Fare**: Tarifa del pasajero
    - **Embarked**: Puerto de embarque (C, Q, S)    
    """
    try:
        # Cargar modelo y metadatos
        model, expected_columns = load_model_and_metadata()
        
        # Convertir datos de entrada a DataFrame
        passenger_dict = passenger.model_dump()
        passenger_df = pd.DataFrame([passenger_dict])
        
        # Aplicar el mismo preprocesamiento que en entrenamiento
        passenger_clean = clean_data(passenger_df)
        passenger_encoded = encode_data(passenger_clean)
        
        # Asegurar que las columnas coincidan con las esperadas por el modelo
        passenger_final = passenger_encoded.reindex(columns=expected_columns, fill_value=0)
        
        #Realizar prediccion
        prediction = model.predict(passenger_final)
        probability = model.predict_proba(passenger_final)
        
        # Preparar respuesta
        survival_prob = probability[0][1]  # Probabilidad de supervivencia
        survived = int(prediction[0])
        
        response = PredictionResponse(
            passenger_id=passenger_dict.get("Name", "unknown"),
            prediction=survived,
            probability=float(survival_prob),
            survival_status="Survived" if survived == 1 else "Did not survive",
            message=f"Probabilidad de supervivencia: {survival_prob:.2%}"
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    
    
    # Ejecutar con: uvicorn src.api:app --reload --host 0.0.0 --port 8000
        
        
        
    
