import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def cargar_datos(file_path):
    df_demanda = pd.read_excel(file_path, sheet_name="Demanda")
    df_stock = pd.read_excel(file_path, sheet_name="Stock")
    return df_demanda, df_stock

def procesar_datos(df_demanda, df_stock):
    demanda_agreg = (
        df_demanda
        .groupby(["producto", "anio", "mes"])['cantidad']
        .sum()
        .reset_index()
        .rename(columns={"cantidad": "demanda_mes"})
    )
    df = pd.merge(df_stock, demanda_agreg, how="left",
                left_on=["material", "anio", "mes"],
                right_on=["producto", "anio", "mes"])
    df = df.drop(columns=["producto"])
    df["demanda_mes"] = df["demanda_mes"].fillna(0)
    df["exceso_logico"] = np.maximum(df["stock"] - df["demanda_mes"] * 1.2, 0)
    return df

def preparar_features(df):
    features = ["anio", "mes", "unidad", "almacen", "sitio", "por_entregar", "por_llegar"]
    X = df[features].copy()
    y = df["exceso_logico"]

    X["por_entregar"] = pd.to_numeric(X["por_entregar"], errors="coerce").fillna(0)
    X["por_llegar"] = pd.to_numeric(X["por_llegar"], errors="coerce").fillna(0)

    num_features = ["anio", "mes", "por_entregar", "por_llegar"]
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