import pandas as pd
import numpy as np

def clean_data (df):
    """
    Realiza la limpieza y transformación inicial del DataFrame.
    Maneja valores faltantes y crea nuevas caracteristicas iniciales.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada crudo.
        
    Returns:
        pd.DataFrame: DataFrame limpio con nuevas caracteristicas
    
    """
    
    # trabajar sobre una copia para evitar modificar el original
    df_clean = df.copy()
    
    # 1. Manejo de valores faltantes  (Usando insights del EDA)
    #---------------------------------------------------------------------------------
    # Imputar 'Age' con la mediana (es robusta a outliners)
    df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
    
    # Imputar los 2 valores faltantes en "Embarked" con la moda (valor más común)
    df_clean["Embarked"] = df_clean["Embarked"].fillna(df_clean['Embarked'].mode()[0])
    
    # Cabin tiene demasiados nulos. Mejor creamos una feature binaria 'Has_Cabin'
    df_clean['Has_Cabin'] = df_clean['Cabin'].notnull().astype(int)
    
    # En el Dataset de test, 'Fare' tiene un valor nulo. Lo imputamos con la mediana.
    if 'Fare' in df_clean.columns:
        df_clean['Fare'] = df_clean['Fare'].fillna(df_clean['Fare'].median())
        
    
    # 2. Ingeniería de Caracteristicas (Feature Engineering)
    #----------------------------------------------------------------------------------
    # Extraer el titulo del nombre (e.g., Mr.,Mrs., Master) será muy informativo
    df_clean['Title'] = df_clean['Name'].str.extract(r'([A-Za-z]+])\. ', expand = False)
    
    # Agrupar titulos raros o menos comunes en una categoría 'Rare'
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Si', 
                   'Jonkheer', 'Dona']
    
    df_clean['Title'] = df_clean['Title'].replace(rare_titles, 'Rare')
    #Consolidad titulos equivalentes
    df_clean['Title'] = df_clean['Title'].replace('Mlle', 'Miss')
    df_clean['Title'] = df_clean['Title'].replace('Ms', 'Miss')
    df_clean['Title'] = df_clean['Title'].replace('Mme', 'Mrs')
    
    # Crear 'FamilySize' combinando SibSp (hermanos/esposo) y Parch (padres/hijos)
    df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1 # +1 para incluir al pasajero
    
    # Crear 'IsAlone' para identificar pasajeros que viajaban solos
    df_clean['IsAlone'] = 0
    df_clean.loc[df_clean['FamilySize'] == 1, 'IsAlone'] = 1 
    
    
    # 3. Eliminar columnas redundantes o que ya no se necesitan
    #-----------------------------------------------------------------------------------
    #'Name' y 'Ticket' son strings complejos y ya extrajimos la información útil (Título)
    # 'Cabin' lo reemplazamos por 'Has_Cabin'
    # 'SinSp' y 'Parch' los reemplazamos por 'FamilySize' e 'IsAlone'
    # 'PassagerID' es un identificador, no es una feature útil para el modelo
    columns_to_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'PassagerID']
    # Usamos errors = 'ignore' por si alguna columna no existe en el dataset de test
    df_clean.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')
    
    return df_clean


def encode_data(df):
    """
    Codifica todas las variables categoricas en numericas para que el modelo pueda usarlas.
    
    Args:
        df (pd.DataFrame): DataFrame limpio con caracteristicas categoricas.
        
    Returns:
        pd.DataFrame: DataFrame con todas las variables en formato numerico
    
    """
    df_encoded = df.copy() 
    
    # Codificacion ordinaria para 'Sex' (asignamos un numero a cada categoria)
    df_encoded['Sex'] = df_encoded['Sex'].map({'male':0, 'female':1})
    
    # Codificacion One-Hot para variables categoricas con mas de dos categorias
    # 'Embarked' (S, C, Q) y 'Title' (Mr, Mrs, Miss, Master, Rare)
    categorical_features = ['Embarked', 'Title']
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_features, prefix=categorical_features)
    
    return df_encoded

#---BLOQUE DE PRUEBA---
# Permite probar el script directamente
if __name__ == '__main__':
    # Cargar datos de prueba
    sample_data = pd.read_csv('/home/mz8k/kaggle-titanic-ml-pipeline/data/train.csv')
    print("DataFrame original: ", sample_data.shape)
    
    # Probar la funcion de limpieza
    cleaned_data = clean_data(sample_data)
    print("DataFrame limpio: ", cleaned_data.shape)
    print("\nColumnas despues de clean_data:\n", cleaned_data.columns.tolist())
    
    # Probar la funcion encoding
    final_data = encode_data(cleaned_data)
    print("\nDataFrame final codificado: ", final_data.shape)
    print("\nColumnas despues de encode_data: ", final_data.tolist())
    print("\nPrimeras 5 filas:\n", final_data.head())
    
