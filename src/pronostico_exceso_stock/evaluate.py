import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.model_selection import learning_curve, cross_val_score, train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from preprocessing import cargar_datos, procesar_datos, preparar_features

# Crear carpeta de salida
os.makedirs("src/pronostico_exceso_stock/graficos", exist_ok=True)

# Cargar y procesar datos
df_demanda, df_stock = cargar_datos("data/datos_limpios.xlsx")
df = procesar_datos(df_demanda, df_stock)
X, y, preprocessor = preparar_features(df)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cargar modelo
modelo = joblib.load("models/modelo_exceso.pkl")
y_pred_stack = modelo.predict(X_test_raw)

# === Curva de aprendizaje ===
train_sizes = np.linspace(0.1, 1.0, 5)
train_sizes, train_scores, test_scores = learning_curve(
    estimator=clone(modelo),
    X=X,
    y=y,
    train_sizes=train_sizes,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

train_rmse = np.sqrt(-train_scores)
test_rmse = np.sqrt(-test_scores)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_rmse.mean(axis=1), 'o-', label="Entrenamiento")
plt.plot(train_sizes, test_rmse.mean(axis=1), 's-', label="Validación")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("RMSE")
plt.title("Curva de Aprendizaje (Modelo Exceso)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("src/pronostico_exceso_stock/graficos/exceso_curva_aprendizaje.png")
plt.close()

# === Curva de pérdida por fold ===
modelos = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoost": GradientBoostingRegressor(random_state=42),
    "KNN": KNeighborsRegressor()
}

X_prepared = modelo.named_steps["preprocessor"].transform(X)

plt.figure(figsize=(8, 6))
for nombre, model in modelos.items():
    rmse_cv = np.sqrt(-cross_val_score(model, X_prepared, y, cv=10, scoring="neg_mean_squared_error"))
    plt.plot(range(1, 11), rmse_cv, marker='o', label=nombre)

plt.xlabel("Fold")
plt.ylabel("RMSE")
plt.title("Curva de Pérdida por Fold (Modelos Base)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("src/pronostico_exceso_stock/graficos/exceso_curva_perdida.png")
plt.close()

# === Curva ROC ===
y_test_bin = (y_test > 0).astype(int)
y_score = y_pred_stack
fpr, tpr, _ = roc_curve(y_test_bin, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC (detección de exceso)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("src/pronostico_exceso_stock/graficos/exceso_curva_roc.png")
plt.close()

# === Exportar productos con exceso de stock predicho ===
X_test_raw = X_test_raw.copy()
X_test_raw["exceso_predicho"] = y_pred_stack

# Reincorporar columnas clave desde df
X_test_raw["material"] = df.loc[X_test_raw.index, "material"]
X_test_raw["anio"] = df.loc[X_test_raw.index, "anio"]
X_test_raw["mes"] = df.loc[X_test_raw.index, "mes"]

# Filtrar casos con exceso positivo
exceso_predicho_df = X_test_raw[X_test_raw["exceso_predicho"] > 0]
exceso_predicho_df = exceso_predicho_df.sort_values(by="exceso_predicho", ascending=False)

# Guardar archivo Excel
os.makedirs("src/pronostico_exceso_stock/salidas", exist_ok=True)
exceso_predicho_df.to_excel("src/pronostico_exceso_stock/salidas/productos_con_exceso.xlsx", index=False)

print("\n📄 Lista de productos con exceso guardada en 'src/pronostico_exceso_stock/salidas/productos_con_exceso.xlsx'")
