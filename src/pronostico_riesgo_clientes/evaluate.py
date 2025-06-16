import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from preprocessing import cargar_datos, procesar_datos, preparar_features

# Crear carpetas
os.makedirs("src/pronostico_riesgo_clientes/graficos", exist_ok=True)
os.makedirs("src/pronostico_riesgo_clientes/salidas", exist_ok=True)

# === Cargar datos ===
df = cargar_datos("data/datos_limpios.xlsx", sheet_name="Demanda")
df = procesar_datos(df)
X, y = preparar_features(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# === Cargar modelo entrenado ===
modelo = joblib.load("models/modelo_riesgo_clientes.pkl")
y_pred = modelo.predict(X_test)

# === Reporte de clasificación ===
reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
reporte_df = pd.DataFrame(reporte).transpose()
reporte_df.to_excel("src/pronostico_riesgo_clientes/salidas/reporte_riesgo_clientes.xlsx")

# === Matriz de confusión ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Riesgo", "Riesgo"], yticklabels=["No Riesgo", "Riesgo"])
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Riesgo Clientes")
plt.tight_layout()
plt.savefig("src/pronostico_riesgo_clientes/graficos/matriz_confusion_riesgo.png")
plt.close()

# === Curva ROC ===
y_proba = modelo.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC - Riesgo Clientes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("src/pronostico_riesgo_clientes/graficos/roc_curve_riesgo.png")
plt.close()

# === Curva Precision-Recall ===
precision, recall, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f"AP = {ap:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall - Riesgo Clientes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("src/pronostico_riesgo_clientes/graficos/precision_recall_curve_riesgo.png")
plt.close()

# === Exportar comparación real vs predicho ===
resultados_df = pd.DataFrame({"Real": y_test, "Predicho": y_pred})
resultados_df.to_excel("src/pronostico_riesgo_clientes/salidas/predicciones_riesgo_clientes.xlsx", index=False)

print("✅ Evaluación completada y resultados exportados.")
