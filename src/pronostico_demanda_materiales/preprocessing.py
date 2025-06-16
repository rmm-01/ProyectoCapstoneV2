import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def cargar_datos(file_path, sheet_name="Stock"):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def limpiar_datos(df):
    columnas_convertir = ['stock', 'en_almacen', 'por_entregar', 'por_llegar', 'ov_1', 'ov_2']
    for col in columnas_convertir:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(" ", "").str.replace(",", ".").astype(float)

    df.dropna(subset=['stock', 'en_almacen', 'anio', 'mes'], inplace=True)

    for col in ['por_entregar', 'por_llegar', 'ov_1', 'ov_2']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df['target'] = df['stock'] - df['en_almacen']
    df = df[df['target'] != 0]
    return df

def preparar_features(df):
    features = ['material', 'anio', 'mes', 'por_entregar', 'por_llegar', 'ov_1', 'ov_2']
    X = df[features].copy()
    y = df['target']
    if X['material'].dtype == 'object':
        le = LabelEncoder()
        X['material'] = le.fit_transform(X['material'])
    return X, y
