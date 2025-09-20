# src/train.py
"""
Titanic ML Training Pipeline - CLI Entry Point
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
import os

# Importar modulos internos
try:
    from src.data_preprocessing import clean_data, encode_data
    from src.model_utils import train_model, evaluate_model
    print("Modulos locales importados correctamente")
except ImportError:
    try:
        from data_preprocessing import clean_data, encode_data
        from model_utils import train_model, evaluate_model
        print("Modulos importados directamente")
    except ImportError as e:
        print(f"Error importando modulos: {e}")
        print("Asegurate de que los archivos existen en src/")
        sys.exit(1)
    
def get_data_paths():
    """
    Detecta automaticamente el entorno y devuelve rutas absolutas correctas.
    """
    # Obtener ruta absoluta del directorio actual
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent.absolute()
    
    # Detectar entorno
    if Path('/app').exists():
        data_dir = Path('/app/data')
        env = 'Docker'
    else:
        data_dir = project_root / 'data'
        env = 'Local'
        
    
    # Definir rutas
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    output_path = data_dir / 'titanic_submission.csv'
    
    print(f"       Entorno: {env}")
    print(f"       Data directory: {data_dir}")
    print(f"       Train path: {train_path}")
    
    return train_path, test_path, output_path

def main():
    print("="*60)
    print("    TITANIC ML PIPELINE - INICIANDO")
    print("="*60)
    
    # 1. Configurar rutas
    print("\n 1/6 - Configurando rutas...")
    train_path, test_path, output_path = get_data_paths()
    
    # 2. Cargar datos
    print("\n 2/6 - Cargando datos...")
    try:    
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f" Train: {train_df.shape}, Test: {test_df.shape}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("   Solucion: Coloca train,csv y test.csv en la carpeta data/")
        return
    
    # 3. Preprocesamiento
    print("\n  3/6 - Preprocesamiento de datos...")
    try:    
        train_clean = clean_data(train_df)
        test_clean = clean_data(test_df)
        train_final = encode_data(train_clean)
        test_final = encode_data(test_clean)
        print(f"   Train final: {train_final.shape}")
        print(f"   Test final: {test_final.shape}")
    except Exception as e:
        print(f" Error en preprocesamiento {e}")
        return
    
    # 4. Preparar datos para entrenamiento
    print("\n4/6 - Preparando datos para entrenamiento...")
    try:
        X = train_final.drop('Survived', axis=1)
        y = train_final['Survived']
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f" X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f" X_val: {X_val.shape}, y_val: {y_val.shape}")
    except Exception as e:
        print(f" Error preparando datos: {e}")
        return
    
    
    # 5. Entrenar modelo
    print("\n 5/6 - Entrenando modelo...")
    try:
        model, params, cv_score = train_model(X_train, y_train)
        print(f" Mejor score (CV): {cv_score:.4f}")
        print(f" Mejores parametros: {params}")
    except Exception as e:
        print(f" Error entrenando modelo: {e}")
        return
    
    # guardar el modelo entrenado para que la API pueda usarlo
    import joblib
    from pathlib import Path
    
    # ...(coddigo existente de entrenamiento)
    
    print("\n Guardando modelo entrenado...")
    try:
        #crear carpeta models si no existe
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # guardar modelo
        model_path = models_dir / 'best_model.pkl'
        joblib.dump(model, model_path)
        print(f" Modelo guardado en: {model_path}")
        
        # Tambien guardar las columnas esperadas para la API
        expected_columns = X_train.columns.tolist()
        joblib.dump(expected_columns, models_dir / 'expected_columns.pkl')
        print(f" Columnas esperadas guardadas en: {len(expected_columns)} columnas")
        
    except Exception as e:
        print(f" Error guardando modelo: {e}")
    return 
    
        
     
    # 6. Evaluar y generar predicciones
    print("\n 6/6 - Generando predicciones...")
    try:
        # Evaluar
        val_accuracy = evaluate_model(model, X_val, y_val) 
        
        # Predecir
        test_predictions = model.predict(test_final)
        
        # Guardar resultados
        submission_df = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': test_predictions
        })
        submission_df.to_csv(output_path, index=False)
        
        # Mostrar estádisticas
        survival_rate = submission_df['Survived'].mean() * 100
        print(f" Accuracy validation: {val_accuracy:.4f}")
        print(f" Tasa de supervivencia predicha: {survival_rate:.1f}%")
        print(f" Submission guardado: {output_path}")
        
    except Exception as e:
        print(f" Error en predicciones: {e}")
        return
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETADA EXITOSAMENTE!")
    print(f" Validation Accuracy: {val_accuracy:.4f}")
    print(f" Cross-Val Score: {cv_score:.4f}")
    print("="*60)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar modelo para competencia Titanic')
    parser.add_argument('--model', type=str, default='gradient_boosting',
                        help='Tipo de modelo a entrenar')
    args = parser.parse_args()
    
    main()

