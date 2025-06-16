import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def cargar_datos(file_path):
    demanda = pd.read_excel(file_path, sheet_name="Demanda")
    stock = pd.read_excel(file_path, sheet_name="Stock")
    return demanda, stock

def procesar_datos(demanda, stock):
    demanda_agg = demanda.groupby(["producto", "anio", "mes"])["cantidad"].sum().reset_index()
    demanda_agg.rename(columns={"cantidad": "demanda_mes"}, inplace=True)

    df = pd.merge(stock, demanda_agg, how="left",
                left_on=["material", "anio", "mes"],
                right_on=["producto", "anio", "mes"])
    df.drop(columns=["producto"], inplace=True)
    df["demanda_mes"] = pd.to_numeric(df["demanda_mes"], errors="coerce").fillna(0)

    for col in ["por_llegar", "por_entregar", "stock", "en_almacen"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    np.random.seed(42)
    df["stock_optimo"] = (
        df["por_entregar"] * np.random.uniform(0.4, 0.6, size=len(df)) +
        df["por_llegar"] * np.random.uniform(0.5, 0.8, size=len(df)) +
        df["stock"] * np.random.uniform(0.1, 0.3, size=len(df)) +
        df["demanda_mes"] * np.random.uniform(0.2, 0.5, size=len(df)) -
        df["en_almacen"] * np.random.uniform(0.3, 0.6, size=len(df))
    )
    df["stock_optimo"] = df["stock_optimo"].clip(lower=0)
    return df

def preparar_features(df):
    features = ["anio", "mes", "unidad", "almacen", "sitio", "por_entregar", "por_llegar", "stock", "en_almacen"]
    X = df[features].copy()
    y = df["stock_optimo"]

    for col in ["unidad", "almacen", "sitio"]:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = SimpleImputer(strategy="median").fit_transform(X)
    return X, y
