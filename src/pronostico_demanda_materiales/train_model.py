import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from preprocessing import cargar_datos, limpiar_datos, preparar_features

# === Cargar y limpiar datos ===
df = cargar_datos("data/datos_limpios.xlsx", sheet_name="Stock")
print("✅ Hoja 'Stock' cargada correctamente")
df = limpiar_datos(df)
print(f"✅ Filas después de limpieza: {len(df)}")

# === Preparar features ===
X, y = preparar_features(df)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Entrenar modelos base ===
modelos = {
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42),
    'GradientBoost': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

print("\n📊 MÉTRICAS DE MODELOS BASE:")
resultados = {}
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    resultados[nombre] = {
        'MAE': mean_absolute_error(y_test, pred),
        'RMSE': mean_squared_error(y_test, pred) ** 0.5,
        'R2': r2_score(y_test, pred)
    }

    print(f"\n🔹 {nombre}")
    for metrica, valor in resultados[nombre].items():
        print(f"{metrica}: {valor:.4f}")

    r2_cv = cross_val_score(modelo, X_scaled, y, cv=5, scoring='r2')
    print(f"R² promedio (CV 5-fold): {r2_cv.mean():.4f} ± {r2_cv.std():.4f}")

# === StackingRegressor completo ===
estimators = [(name, modelos[name]) for name in modelos]
meta_model = RidgeCV()
stack_model = StackingRegressor(estimators=estimators, final_estimator=meta_model, cv=5)
stack_model.fit(X_train, y_train)

# Evaluación del metamodelo
y_meta_pred = stack_model.predict(X_test)
print("\n🔗 MÉTRICAS DEL METAMODELO (Stacking):")
print(f"MAE: {mean_absolute_error(y_test, y_meta_pred):.4f}")
print(f"RMSE: {mean_squared_error(y_test, y_meta_pred) ** 0.5:.4f}")
print(f"R2: {r2_score(y_test, y_meta_pred):.4f}")

r2_cv_meta = cross_val_score(stack_model, X_scaled, y, cv=5, scoring='r2')
print(f"\n🔁 R² promedio del Metamodelo (CV): {r2_cv_meta.mean():.4f} ± {r2_cv_meta.std():.4f}")

# Guardar el modelo completo con escalador en un pipeline
modelo_final = Pipeline([
    ("scaler", scaler),
    ("stacking", stack_model)
])

os.makedirs("models", exist_ok=True)
joblib.dump(modelo_final, "models/modelo_materiales.pkl")
print("\n🗓 Modelo guardado en 'models/modelo_materiales.pkl'")
