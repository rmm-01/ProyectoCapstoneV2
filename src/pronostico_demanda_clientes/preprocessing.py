import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def cargar_datos(file_path):
    df_demanda = pd.read_excel(file_path, sheet_name="Demanda")
    df_stock = pd.read_excel(file_path, sheet_name="Stock")
    return df_demanda, df_stock

def procesar_datos(df_demanda, df_stock):
    df_demanda = df_demanda.dropna(subset=['cliente_id', 'cliente', 'producto', 'cantidad', 'anio', 'mes'])
    df_demanda = df_demanda[df_demanda['cantidad'] > 0]

    le_cliente = LabelEncoder()
    le_producto = LabelEncoder()

    df_demanda['cliente_id'] = le_cliente.fit_transform(df_demanda['cliente_id'].astype(str))
    df_demanda['producto'] = le_producto.fit_transform(df_demanda['producto'].astype(str))

    df_demanda['media_cliente'] = df_demanda.groupby('cliente_id')['cantidad'].transform('mean')
    df_demanda['media_producto'] = df_demanda.groupby('producto')['cantidad'].transform('mean')
    df_demanda['media_mes'] = df_demanda.groupby('mes')['cantidad'].transform('mean')
    df_demanda['media_cliente_producto'] = df_demanda.groupby(['cliente_id', 'producto'])['cantidad'].transform('mean')

    df_stock['producto'] = df_stock['material'].astype(str).map(dict(zip(le_producto.classes_, le_producto.transform(le_producto.classes_))))
    df_stock = df_stock.dropna(subset=['producto'])
    df_stock['producto'] = df_stock['producto'].astype(int)

    stock_agg = df_stock.groupby(['producto', 'anio', 'mes'])[['por_entregar', 'stock', 'en_almacen']].mean().reset_index()
    df = pd.merge(df_demanda, stock_agg, on=['producto', 'anio', 'mes'], how='left')
    df.fillna(0, inplace=True)

    return df

def preparar_features(df):
    X = df[['cliente_id', 'producto', 'anio', 'mes', 'media_cliente', 'media_producto',
            'media_mes', 'media_cliente_producto', 'por_entregar', 'stock', 'en_almacen']]
    y = df['cantidad']
    return X, y, df['cliente_id']
