import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import learning_curve, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from preprocessing import cargar_datos, limpiar_datos, preparar_features
from sklearn.preprocessing import StandardScaler

# === Preparar entorno ===
os.makedirs("src/pronostico_demanda_materiales/graficos", exist_ok=True)
os.makedirs("src/pronostico_demanda_materiales/salidas", exist_ok=True)

# === Cargar y preparar datos ===
df = cargar_datos("data/datos_limpios.xlsx", sheet_name="Stock")
df = limpiar_datos(df)
X, y = preparar_features(df)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === División de datos ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Cargar modelo entrenado ===
modelo = joblib.load("models/modelo_materiales.pkl")

# === Curva de aprendizaje ===
train_sizes = np.linspace(0.1, 1.0, 5)
train_sizes, train_scores, test_scores = learning_curve(
    estimator=clone(modelo),
    X=X_scaled, y=y,
    train_sizes=train_sizes, cv=5, scoring='r2', n_jobs=-1
)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label="Entrenamiento")
plt.plot(train_sizes, test_scores.mean(axis=1), 's-', label="Validación")
plt.title("Curva de Aprendizaje (Materiales)")
plt.xlabel("Tamaño de entrenamiento")
plt.ylabel("R²")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("src/pronostico_demanda_materiales/graficos/curva_aprendizaje_materiales.png")
plt.close()

# === Curva de pérdida ===
modelos_base = {
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42),
    "GradientBoost": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}
plt.figure(figsize=(8, 6))
for nombre, model in modelos_base.items():
    rmse_cv = np.sqrt(-cross_val_score(model, X_scaled, y, cv=5, scoring="neg_mean_squared_error"))
    plt.plot(range(1, 6), rmse_cv, marker='o', label=nombre)
plt.xlabel("Fold")
plt.ylabel("RMSE")
plt.title("Curva de Pérdida por Fold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("src/pronostico_demanda_materiales/graficos/curva_perdida_materiales.png")
plt.close()

# === Gráfico Real vs Predicho ===
y_pred = modelo.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Real")
plt.ylabel("Predicho")
plt.title("Real vs Predicho (Materiales)")
plt.grid(True)
plt.tight_layout()
plt.savefig("src/pronostico_demanda_materiales/graficos/real_vs_predicho_materiales.png")
plt.close()

# === Exportar resultados ===
pred_df = pd.DataFrame({"Real": y_test.values, "Predicho": y_pred})
pred_df.to_excel("src/pronostico_demanda_materiales/salidas/predicciones_materiales.xlsx", index=False)

# === Distribución Real Lineal ===
plt.figure(figsize=(8, 6))
plt.hist(y, bins=50, color='lightgray', edgecolor='black', alpha=0.8, density=True)
sns.kdeplot(y, color='blue', linewidth=2)
plt.xlabel("Demanda Real")
plt.ylabel("Frecuencia")
plt.title("Distribución de Demanda Real - Materiales")
plt.tight_layout()
plt.savefig("src/pronostico_demanda_materiales/graficos/distribucion_demanda_materiales.png")
plt.close()

# === Distribución Real Logarítmica ===
y_log = np.log1p(y)
plt.figure(figsize=(8, 6))
plt.hist(y_log, bins=50, color='lightgray', edgecolor='black', alpha=0.8, density=True)
sns.kdeplot(y_log, color='blue', linewidth=2)
plt.xlabel("Log(1 + Demanda)")
plt.ylabel("Frecuencia")
plt.title("Distribución Logarítmica de Demanda - Materiales")
plt.tight_layout()
plt.savefig("src/pronostico_demanda_materiales/graficos/distribucion_demanda_materiales_log.png")
plt.close()




print("✅ Evaluación completada y gráficas + predicciones guardadas.")
