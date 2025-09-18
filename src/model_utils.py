"""_
    _Modulo con las funciones para entrenar y evaluar el modelo._
    """
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def train_model(X_train, y_train):
    """
    Entrena un modelo de Gradient Boosting con validacion cruzada para optimizar hiperparametros.

    Args:
        X_train (pd.DataFrame): Features de entrenamiento.
        y_train (pd.Series): Target de entrenamiento.
        
    Returns:
        tuple: (mejor modelo, mejores hiperparametros, mejor score)
    """
    print(" Entrenando modelo (GrindSearchCV)...")
    
    # Definir el modeloy los hiperparametros a probar
    model = GradientBoostingClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 4],
    }
    
    # Configurar GridSearchCV
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid,
        cv=3,               # 3-fold cross-validation   
        scoring='accuracy', # Metric to optimize
        n_jobs=-1,         # Usar todos los cores disponibles
        verbose=1
    )   
    
    # ejecutar la busqueda
    grid_search.fit(X_train, y_train)
    
    print(f"mejores hiperparametros encontrados: {grid_search.best_params_}")
    print(f"mejor score en validacion cruzada: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def evaluate_model(model, X_val, y_val):
    """
    Evalua el modelo en el conjunto de validacion y muestra metricas de rendimiento.    
    """
    print(" Evaluando modelo en el conjunto de validacion...")
    
    # predecir con el modelo optimizado en el conjunto de validacion
    y_val_pred = model.predict(X_val)
    
    # calcular metricas de evaluacion final
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"Accuracy final en el conjunto de validacion: {val_accuracy:.4f}")
    print("\nReporte de Clasificacion:")
    print(classification_report(y_val, y_val_pred))
    
    return val_accuracy