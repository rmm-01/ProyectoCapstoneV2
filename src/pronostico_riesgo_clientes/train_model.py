import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from preprocessing import cargar_datos, procesar_datos, preparar_features

# === Cargar y procesar datos ===
df = cargar_datos("data/datos_limpios.xlsx", sheet_name="Demanda")
df = procesar_datos(df)
X, y = preparar_features(df)

# === División de datos ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# === Definir modelos base con regularización para evitar overfitting ===
rf = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
et = ExtraTreesClassifier(n_estimators=30, max_depth=3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=30, max_depth=2, learning_rate=0.1, random_state=42)

print("\n📊 MÉTRICAS DE MODELOS BASE:")
for name, model in [('RandomForest', rf), ('ExtraTrees', et), ('GradientBoosting', gb)]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n🔹 {name}")
    print(classification_report(y_test, y_pred, zero_division=1))

# === Modelo Stacking ===
stack_model = StackingClassifier(
    estimators=[('rf', rf), ('et', et), ('gb', gb)],
    final_estimator=LogisticRegression(),
    cv=3
)
stack_model.fit(X_train, y_train)
y_stack_pred = stack_model.predict(X_test)

print("\n🔸 MÉTRICAS DEL METAMODELO (Stacking):")
print(classification_report(y_test, y_stack_pred, zero_division=1))

# === Guardar modelo entrenado ===
os.makedirs("models", exist_ok=True)
joblib.dump(stack_model, "models/modelo_riesgo_clientes.pkl")
print("\n✅ Modelo guardado como 'models/modelo_riesgo_clientes.pkl'")
