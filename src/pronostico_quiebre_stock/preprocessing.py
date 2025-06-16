import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def cargar_datos(file_path):
    # Cargar hoja de stock únicamente
    df_stock = pd.read_excel(file_path, sheet_name="Stock")
    return None, df_stock

def procesar_datos(_, df_stock):
    # Asegurar columnas numéricas
    df_stock["stock"] = pd.to_numeric(df_stock["stock"], errors="coerce").fillna(0)

    # Inicializar columna de demanda simulada
    df_stock["demanda_mes"] = 0

    # Simular quiebre en 10% de registros aleatorios
    muestra = df_stock.sample(frac=0.10, random_state=42)
    df_stock.loc[muestra.index, "stock"] = 0
    df_stock.loc[muestra.index, "demanda_mes"] = 100

    # Calcular variable objetivo
    df_stock["quiebre_logico"] = np.maximum((df_stock["demanda_mes"] * 1.5) - df_stock["stock"], 0)

    return df_stock

def preparar_features(df):
    # Usamos demanda_mes como una feature importante
    features = ["anio", "mes", "unidad", "almacen", "sitio", "por_entregar", "demanda_mes"]
    X = df[features].copy()
    y = df["quiebre_logico"]

    # Conversión segura
    X["por_entregar"] = pd.to_numeric(X["por_entregar"], errors="coerce").fillna(0)
    X["demanda_mes"] = pd.to_numeric(X["demanda_mes"], errors="coerce").fillna(0)

    # Variables numéricas y categóricas
    num_features = ["anio", "mes", "por_entregar", "demanda_mes"]
    cat_features = ["unidad", "almacen", "sitio"]

    numeric_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transform, num_features),
        ("cat", categorical_transform, cat_features)
    ])

    return X, y, preprocessor