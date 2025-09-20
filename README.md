# 🚢 Titanic ML Pipeline - End-to-End Machine Learning Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![GitHub Actions CI/CD](https://img.shields.io/badge/CI/CD-GitHub_Actions-2088FF?logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask API](https://img.shields.io/badge/API-Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Pytest](https://img.shields.io/badge/Testing-Pytest-0A9EDC?logo=python&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Tabla de Contenidos
- [🚀 Inicio Rápido](#-inicio-rápido)
- [🏃‍♂️ Guía de Ejecución Paso a Paso](#-guía-de-ejecución-paso-a-paso)
- [🐳 Ejecución con Docker](#-ejecución-con-docker)
- [📊 Verificación y Monitoreo](#-verificación-y-monitoreo)
- [🔧 Operaciones Avanzadas](#-operaciones-avanzadas)
- [🆘 Solución de Problemas](#-solución-de-problemas)
- [📁 Estructura del Proyecto](#-estructura-del-proyecto)

## 🚀 Inicio Rápido

### Prerrequisitos
```bash
# Solo necesitas:
python --version  # Python 3.8+
git --version     # Git

# Clonar el repositorio
git clone https://github.com/D4n0nin0/kaggle-titanic-ml-pipeline.git
cd kaggle-titanic-ml-pipeline

# Crear entorno virtual (Linux/Mac)
python -m venv venv
source venv/bin/activate

# Windows:
# python -m venv venv
# venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt


# Paso 1 VERIFICAR INSTALACION

# Verificar versión de Python
python --version

# Verificar que todas las dependencias se instalaron correctamente
python -c "import sklearn; import flask; import pandas as pd; print('✅ Todas las dependencias instaladas correctamente!')"

# Paso 2 VERIFICAR LA ESTRUCTURA DEL PROYECTO

# Ver organización completa del proyecto
ls -la

# Ver contenido del directorio data/
ls data/

# Ver código fuente
ls src/

# Ver archivos de tests
ls tests/


# Paso 3 PREPROCESAMIENTO DE DATOS

# Ejecutar preprocesamiento de datos
python -c "
from src.data_preprocessing import preprocess_data
preprocess_data('data/train.csv', 'data/processed_train.csv')
print('✅ Preprocesamiento de datos completado exitosamente!')
print('📊 Datos procesados guardados en: data/processed_train.csv')
"


# Paso 4 ENTRENAMIENTO DEL MODELO DE ML

# Entrenar el modelo (creará archivos en models/)
python src/train.py

# Verificar que se crearon los archivos del modelo
ls models/
# Deberías ver: random_forest_titanic.pkl y model_metrics.json


# Paso 5  EJECUTAR TESTS 

# Ejecutar todos los tests con output verbose
python -m pytest tests/ -v

# Ejecutar tests con reporte de cobertura
python -m pytest --cov=src tests/ --cov-report=term-missing

# Ejecutar tests específicos
python -m pytest tests/test_api.py -v

# Paso 6 INICIAR EL API DE PREDICCIONES

# Iniciar Flask API (ejecutará en http://localhost:5000)
python src/api.py

# Mantén esta terminal abierta - el API seguirá ejecutándose
# Abre una nueva terminal para los siguientes pasos

# Paso 7 PROBAR ENDPOINT DE HEALTH DEL API

# Probar endpoint de health (en nueva terminal)
curl http://localhost:5000/health

# Respuesta esperada:
# {"status": "healthy", "model_loaded": true, "version": "1.0"}

#PASO 8 PROBAR ENDPOINT DE PREDICCIONES  CON DIFERENTES ESCENARIOS

# Caso de prueba 1: Mujer primera clase (alta probabilidad de supervivencia)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 1,
    "Sex": "female", 
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 50,
    "Embarked": "C"
  }'

# Caso de prueba 2: Hombre tercera clase (baja probabilidad de supervivencia)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 3,
    "Sex": "male", 
    "Age": 30,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.5,
    "Embarked": "S"
  }'

# Caso de prueba 3: Grupo familiar
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 2,
    "Sex": "female", 
    "Age": 28,
    "SibSp": 1,
    "Parch": 2,
    "Fare": 25,
    "Embarked": "Q"
  }'


# Paso 9 EXPLORAR JUPYTER NOTEBOOKS PARA ANALISIS

# Instalar Jupyter si no está instalado
pip install jupyter

# Iniciar servidor de Jupyter notebook
jupyter notebook

# En la interfaz web, abrir y ejecutar:
# 1. notebooks/01_eda.ipynb (Análisis Exploratorio de Datos)
# 2. notebooks/02_data_preprocessing.ipynb 
# 3. notebooks/03_model_training.ipynb

# Paso 10 EJECUTAR PIPELINE COMPLETO

# Ejecutar todo el pipeline de principio a fin
python src/train.py && python -m pytest tests/ -v && python src/api.py

```
# 🐳 Ejecución con Docker

```bash

# PASO 1 CONSTRUIR IMAGEN DE DOCKER

# Construir la imagen de Docker con tag 'titanic-ml'
docker build -t titanic-ml .

# Verificar que la imagen fue creada
docker images | grep titanic-ml

#PASO 2 EJECUTAR CONTENEDOR DE DOCKER

# Ejecutar el contenedor mapeando puerto 5000
docker run -p 5000:5000 --name titanic-container titanic-ml

# Ejecutar en modo detached (segundo plano)
# docker run -d -p 5000:5000 --name titanic-container titanic-ml

# PASO 3 PROBAR API DOCKERIZADO

# Probar endpoint de health
curl http://localhost:5000/health

# Probar endpoint de predicciones
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 1,
    "Sex": "female", 
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 50,
    "Embarked": "C"
  }'

# PASO 4 GESTIONAR CONTENEDOR DE DOCKER

# Detener el contenedor
docker stop titanic-container

# Iniciar el contenedor nuevamente
docker start titanic-container

# Ver logs del contenedor
docker logs titanic-container

# Eliminar contenedor
docker rm titanic-container

# Eliminar imagen
docker rmi titanic-ml

```
# 📊 Verificación y Monitoreo

```bash

# VERIFICAR METRICAS DEL MODELO

# Verificar archivo de métricas del modelo
cat models/model_metrics.json

# Output esperado:
# {"accuracy": 0.82, "precision": 0.85, "recall": 0.78, "f1_score": 0.81}

# PROBAR API CON script de python

# Crear archivo test_api.py
echo '
import requests
import json

# Probar endpoint de health
response = requests.get("http://localhost:5000/health")
print("Health check:", response.json())

# Probar predicción
data = {
    "Pclass": 1,
    "Sex": "female", 
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 50,
    "Embarked": "C"
}

response = requests.post("http://localhost:5000/predict", json=data)
print("Prediction result:", response.json())
' > test_api.py

# Ejecutar script de prueba
python test_api.py

# VERIFICAR ESTADO DEL PIPELINE CI/CD

# Visitar GitHub repository → pestaña Actions