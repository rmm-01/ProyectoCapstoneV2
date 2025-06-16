import pandas as pd
import unicodedata
from datetime import datetime

# --------------------------------------
# FUNCIONES DE UTILIDAD
# --------------------------------------

def estandarizar_columnas(columnas):
    return [
        unicodedata.normalize('NFKD', c).encode('ascii', 'ignore').decode('utf-8').lower().strip().replace(" ", "_")
        for c in columnas
    ]

def normalizar_texto(texto):
    if isinstance(texto, str):
        return unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8').lower().strip()
    return texto

def categorizar_producto(texto):
    texto = str(texto).lower()
    if "geomalla" in texto:
        return "Geomalla"
    elif "geodren" in texto:
        return "Geodren"
    elif "flexilona" in texto:
        return "Flexilona"
    elif "manta" in texto:
        return "Manta"
    elif "tanque" in texto:
        return "Tanque"
    elif "gabion" in texto or "gavion" in texto:
        return "Gavión"
    elif "tubería" in texto or "tuberia" in texto:
        return "Tubería"
    else:
        return "Otro"

# --------------------------------------
# LIMPIEZA Y UNIFICACIÓN DE DEMANDA (2020 + 2024)
# --------------------------------------

def limpiar_data_demanda_total(archivo_2020, archivo_2024):
    try:
        df_2024 = pd.read_excel(archivo_2024, sheet_name="Data", usecols=[
            'Cliente', 'Monto S/.', 'Facturación', 'Oportunidad / Descripción'
        ])
        df_2024 = df_2024.dropna(subset=['Cliente', 'Monto S/.', 'Facturación'])

        df_2024['anio'] = pd.to_datetime(df_2024['Facturación']).dt.year
        df_2024['mes'] = pd.to_datetime(df_2024['Facturación']).dt.month

        df_2024 = df_2024.rename(columns={
            'Cliente': 'cliente',
            'Monto S/.': 'monto',
            'Oportunidad / Descripción': 'producto'
        })

        # 👉 Categorización de productos 2024
        df_2024['producto'] = df_2024['producto'].astype(str).apply(categorizar_producto)

        df_2024['cliente_id'] = df_2024['cliente'].astype('category').cat.codes
        df_2024['cliente_id'] = df_2024['cliente_id'].apply(lambda x: f"cliente_{x:03d}")

        df_2024['cantidad'] = df_2024['monto'] / 1000
        df_2024['cantidad'] = pd.to_numeric(df_2024['cantidad'], errors='coerce')
        df_2024 = df_2024[df_2024['cantidad'] > 0]
        df_2024 = df_2024.dropna(subset=['producto', 'monto', 'cantidad'])

        relaciones_precio = df_2024.groupby(['cliente', 'producto']) \
            .apply(lambda x: x['monto'].sum() / x['cantidad'].sum()).to_dict()
        relaciones_producto = df_2024.groupby('producto') \
            .apply(lambda x: x['monto'].sum() / x['cantidad'].sum()).to_dict()
        precio_general = df_2024['monto'].sum() / df_2024['cantidad'].sum()

        df_2024['ejecutivo'] = None
        columnas_finales = ['cliente_id', 'cliente', 'producto', 'anio', 'mes', 'cantidad', 'monto', 'ejecutivo']
        df_2024 = df_2024[columnas_finales]

        # === 2020 ===
        xls_2020 = pd.ExcelFile(archivo_2020)
        hojas = xls_2020.sheet_names
        datos_2020 = []

        for hoja in hojas:
            df = pd.read_excel(archivo_2020, sheet_name=hoja, header=None)
            if df.shape[0] <= 3:
                continue

            df.columns = df.iloc[3]
            df = df.iloc[4:]
            if 'Cliente' not in df.columns or 'Ejecutivo de Ventas' not in df.columns:
                continue

            df = df.rename(columns={'Cliente': 'cliente', 'Ejecutivo de Ventas': 'ejecutivo'})
            df = df[df['cliente'].notna()]
            columnas_anio = [col for col in df.columns if str(col).startswith('202')]

            df_long = df.melt(id_vars=['cliente', 'ejecutivo'], value_vars=columnas_anio,
                              var_name='anio', value_name='monto')

            df_long['producto'] = hoja  # ya coincide con categorías (Geomalla, Geodren, etc.)
            df_long['anio'] = pd.to_numeric(df_long['anio'], errors='coerce')
            df_long['monto'] = pd.to_numeric(df_long['monto'], errors='coerce')
            df_long = df_long.dropna(subset=['monto'])

            df_long['cliente_id'] = df_long['cliente'].astype('category').cat.codes
            df_long['cliente_id'] = df_long['cliente_id'].apply(lambda x: f"cliente_{x:03d}")
            df_long['mes'] = 7

            def estimar_cantidad(row):
                clave = (row['cliente'], row['producto'])
                if clave in relaciones_precio:
                    return row['monto'] / relaciones_precio[clave]
                elif row['producto'] in relaciones_producto:
                    return row['monto'] / relaciones_producto[row['producto']]
                else:
                    return row['monto'] / precio_general

            df_long['cantidad'] = df_long.apply(estimar_cantidad, axis=1)
            df_final = df_long[columnas_finales]
            datos_2020.append(df_final)

        df_2020 = pd.concat(datos_2020, ignore_index=True)
        df_total = pd.concat([df_2020, df_2024], ignore_index=True)

        print("🔍 Diagnóstico DEMANDA unificada:")
        print(df_total[['cliente_id', 'monto', 'cantidad', 'anio']].isnull().sum())
        print(f"✅ Total filas DEMANDA: {len(df_total)}")
        return df_total

    except Exception as e:
        raise RuntimeError(f"❌ Error en limpieza de demanda TOTAL: {str(e)}")

# --------------------------------------
# LIMPIEZA DE STOCK (solo 2024)
# --------------------------------------

def limpiar_data_stock(df):
    try:
        df = df.copy()
        df.columns = estandarizar_columnas(df.columns)

        columnas_renombrar = {
            'codigo_de_articulo': 'material',
            'disponible': 'stock'
        }
        df = df.rename(columns={col: columnas_renombrar[col] for col in columnas_renombrar if col in df.columns})

        df['stock'] = pd.to_numeric(df.get('stock'), errors='coerce')

        df['anio'] = 2024
        df['mes'] = 4

        df['material'] = df['material'].apply(normalizar_texto)
        df = df.drop_duplicates(subset=['material', 'anio', 'stock'])
        df = df[df['stock'] > 0]

        print("🔍 Diagnóstico stock original:")
        print(df[['material', 'stock', 'anio']].isnull().sum())
        print(f"✅ Total filas STOCK base: {len(df)}")
        return df

    except Exception as e:
        raise RuntimeError(f"❌ Error en limpieza de stock: {str(e)}")

# --------------------------------------
# BLOQUE DE PRUEBA DIRECTA
# --------------------------------------

if __name__ == "__main__":
    try:
        print("\n🧪 Ejecutando limpieza de DEMANDA...")
        archivo_2020 = "Copia de BASE COMERCIAL JULIO 2020.xlsx"
        archivo_2024 = "Copia de BASES DE DATOS GENERAL 2024 (005).xlsx"
        df_demanda = limpiar_data_demanda_total(archivo_2020, archivo_2024)

        print("\n🧪 Ejecutando limpieza de STOCK...")
        archivo_stock = "Copia de Stock hasta 02.04.xlsx"
        hoja_stock = "General"
        df_stock = pd.read_excel(archivo_stock, sheet_name=hoja_stock)
        df_stock_limpio = limpiar_data_stock(df_stock)

        # Simulación stock 2020–2023
        factores = {2020: 0.7, 2021: 0.8, 2022: 0.9, 2023: 1.0}
        stock_simulados = []

        for anio, factor in factores.items():
            df_sim = df_stock_limpio.copy()
            df_sim["anio"] = anio
            df_sim["mes"] = 7
            df_sim["stock"] = (df_sim["stock"] * factor).round(0).astype(int)
            stock_simulados.append(df_sim)

        stock_simulados.append(df_stock_limpio.copy())  # año 2024
        df_stock_multianual = pd.concat(stock_simulados, ignore_index=True)

        with pd.ExcelWriter("datos_limpios.xlsx", engine="openpyxl") as writer:
            df_demanda.to_excel(writer, sheet_name="Demanda", index=False)
            df_stock_multianual.to_excel(writer, sheet_name="Stock", index=False)

        print("✅ Archivo 'datos_limpios.xlsx' generado con hojas 'Demanda' y 'Stock' (multianual incluido)")

    except Exception as e:
        print(f"❌ Error en ejecución principal: {str(e)}")
