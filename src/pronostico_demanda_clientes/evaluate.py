import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from preprocessing import cargar_datos, procesar_datos, preparar_features
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import learning_curve
import numpy as np

# === Preparar entorno ===
os.makedirs("src/pronostico_demanda_clientes/graficos", exist_ok=True)
os.makedirs("src/pronostico_demanda_clientes/resultados", exist_ok=True)

# === Cargar modelo entrenado ===
modelo = joblib.load("models/modelo_clientes.pkl")

# === Cargar y preparar datos ===
demanda, stock = cargar_datos("data/datos_limpios.xlsx")
df = procesar_datos(demanda, stock)
X, y, grupos = preparar_features(df)

# === División por grupo ===
split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, test_idx in split.split(X, y, grupos):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# === Predicción ===
y_pred = modelo.predict(X_test)

# === Métricas ===
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n📈 Evaluación del modelo Stacking:")
print(f"  MAE:  {mae:.2f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  R²:   {r2:.4f}")


# === Gráfico Real vs Predicho ===
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Cantidad Real")
plt.ylabel("Cantidad Predicha")
plt.title("Predicción del Metamodelo vs Valores Reales")
plt.grid(True)
plt.tight_layout()
plt.savefig("src/pronostico_demanda_clientes/graficos/clientes_real_vs_predicho.png")
plt.close()

# === Curva de Aprendizaje ===
train_sizes, train_scores, test_scores = learning_curve(
    estimator=modelo, X=X, y=y, cv=5, scoring="r2", n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42
)
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', label="Entrenamiento")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Validación")
plt.xlabel("Cantidad de Datos de Entrenamiento")
plt.ylabel("R²")
plt.title("Curva de Aprendizaje - Modelo Stacking")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig("src/pronostico_demanda_clientes/graficos/curva_aprendizaje_clientes.png")
plt.close()

# === Curva de Pérdida ===
plt.figure(figsize=(8, 6))
errores = np.abs(y_test - y_pred)
plt.plot(range(len(errores)), errores, alpha=0.7)
plt.title("Curva de Pérdida Absoluta - Clientes")
plt.xlabel("Índice de Prueba")
plt.ylabel("Error Absoluto")
plt.grid(True)
plt.tight_layout()
plt.savefig("src/pronostico_demanda_clientes/graficos/curva_perdida_clientes.png")
plt.close()

print("✅ Todos los gráficos generados correctamente.")


import seaborn as sns
import numpy as np

# === Histograma original ===
plt.figure(figsize=(8, 6))
sns.histplot(y_test, kde=True, color="gray", edgecolor="black")
plt.title("Distribución de Demanda Real - Clientes")
plt.xlabel("Demanda")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("src/pronostico_demanda_clientes/graficos/distribucion_demanda_real_clientes.png")
plt.close()

# === Histograma logarítmico (log1p para evitar log(0)) ===
plt.figure(figsize=(8, 6))
sns.histplot(np.log1p(y_test), kde=True, color="steelblue", edgecolor="black")
plt.title("Distribución Logarítmica de Demanda Real - Clientes")
plt.xlabel("log(1 + Demanda)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("src/pronostico_demanda_clientes/graficos/distribucion_log_demandas_clientes.png")
plt.close()
