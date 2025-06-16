import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

def cargar_datos(file_path, sheet_name="Demanda"):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def procesar_datos(df):
    # Crear etiqueta binaria 'riesgo'
    productos_riesgo = ['SolidMat', 'Sedimentadores', 'Biomanto']
    df['riesgo'] = df['producto'].isin(productos_riesgo).astype(int)
    return df

def preparar_features(df):
    # Separar positivos y negativos
    positivos = df[df['riesgo'] == 1]
    negativos = df[df['riesgo'] == 0]

    # Resamplear con reemplazo y ruido leve
    positivos_aug = resample(positivos, replace=True, n_samples=180, random_state=42)
    negativos_sample = resample(negativos, n_samples=300, random_state=42)

    positivos_aug['cantidad'] += np.random.randint(-3, 3, size=positivos_aug.shape[0])
    positivos_aug['monto'] += np.random.normal(0, 200, size=positivos_aug.shape[0])
    positivos_aug['mes'] = (positivos_aug['mes'] + np.random.randint(-2, 3, size=positivos_aug.shape[0])).clip(1, 12)

    df_sintetico = pd.concat([negativos_sample, positivos_aug]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Preparar X e y
    X = df_sintetico[['cliente_id', 'producto', 'anio', 'mes', 'cantidad', 'monto']].copy()
    X['cliente_id'] = LabelEncoder().fit_transform(X['cliente_id'])
    X['producto'] = LabelEncoder().fit_transform(X['producto'])
    y = df_sintetico['riesgo']

    return X, y
