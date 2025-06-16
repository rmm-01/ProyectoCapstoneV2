import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from preprocessing import cargar_datos, procesar_datos, preparar_features

print("\n📊 Ejecutando evaluación del módulo 'stock_optimo'...")

# === Cargar modelo y datos ===
modelo = joblib.load("models/modelo_stock_optimo.pkl")
demanda, stock = cargar_datos("data/datos_limpios.xlsx")
df = procesar_datos(demanda, stock)
X, y = preparar_features(df)

# === Predicción ===
y_pred = modelo.predict(X)

# === Exportar comparación ===
os.makedirs("src/pronostico_stock_optimo/resultados", exist_ok=True)
pd.DataFrame({"Real": y, "Predicho": y_pred}).to_excel(
    "src/pronostico_stock_optimo/resultados/predicciones_stock_optimo.xlsx", index=False
)

# === Gráfico Real vs Predicho ===
os.makedirs("src/pronostico_stock_optimo/graficos", exist_ok=True)
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.6, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Stock Óptimo Real")
plt.ylabel("Predicción")
plt.title("Predicción del Stock Óptimo vs Real")
plt.grid(True)
plt.tight_layout()
plt.savefig("src/pronostico_stock_optimo/graficos/stock_optimo_real_vs_predicho.png")
plt.close()

# === Curva de Aprendizaje ===
train_sizes, train_scores, test_scores = learning_curve(
    modelo, X, y, cv=5, scoring="neg_root_mean_squared_error",
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1, shuffle=True, random_state=42
)
train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', label="Entrenamiento")
plt.plot(train_sizes, test_scores_mean, 's-', label="Validación")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("RMSE")
plt.title("Curva de Aprendizaje (Modelo Stock Óptimo)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("src/pronostico_stock_optimo/graficos/curva_aprendizaje_stock_optimo.png")
plt.close()

# === Curva de Pérdida por modelo base ===
modelos_base = {
    "RandomForest": RandomForestRegressor(n_estimators=60, max_depth=6, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=60, max_depth=6, random_state=42),
    "GradientBoost": GradientBoostingRegressor(n_estimators=60, max_depth=4, random_state=42),
}

plt.figure(figsize=(8, 6))
for nombre, modelo_base in modelos_base.items():
    rmse_scores = np.sqrt(-cross_val_score(modelo_base, X, y, cv=5, scoring="neg_mean_squared_error"))
    plt.plot(range(1, 6), rmse_scores, marker='o', label=nombre)
plt.xlabel("Fold")
plt.ylabel("RMSE")
plt.title("Curva de Pérdida por Fold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("src/pronostico_stock_optimo/graficos/curva_perdida_stock_optimo.png")
plt.close()

# === Gráfico de Distribución del Target ===
plt.figure(figsize=(8, 6))
plt.hist(y, bins=50, color='gray', edgecolor='black', alpha=0.6, density=True)
y_range = np.linspace(y.min(), y.max(), 500)
from scipy.stats import gaussian_kde
kde = gaussian_kde(y)
plt.plot(y_range, kde(y_range), color='blue')
plt.title("Distribución del Stock Óptimo Real")
plt.xlabel("Stock Óptimo")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("src/pronostico_stock_optimo/graficos/distribucion_stock_optimo.png")
plt.close()

print("✅ Evaluación y gráficos completados correctamente.")
