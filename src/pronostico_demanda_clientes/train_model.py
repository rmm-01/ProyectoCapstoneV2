import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from preprocessing import cargar_datos, procesar_datos, preparar_features

# === Cargar y procesar datos ===
demanda, stock = cargar_datos("data/datos_limpios.xlsx")
df = procesar_datos(demanda, stock)
X, y, grupos = preparar_features(df)

# === División por grupo (cliente) ===
splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, test_idx in splitter.split(X, y, grupos):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# === Modelos base ===
modelos = {
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'GradientBoost': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42)
}

print("\n📊 MÉTRICAS DE MODELOS BASE:")
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    print(f"\n🔹 {nombre}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")

# === Metamodelo y Stacking ===
stack = StackingRegressor(
    estimators=[
        ('rf', modelos['RandomForest']),
        ('gb', modelos['GradientBoost']),
        ('et', modelos['ExtraTrees'])
    ],
    final_estimator=RidgeCV(),
    cv=5
)

stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)

print("\n🔸 MÉTRICAS DEL MODELO STACKING (FINAL):")
print(f"  MAE:  {mean_absolute_error(y_test, y_pred_stack):.2f}")
print(f"  RMSE: {mean_squared_error(y_test, y_pred_stack) ** 0.5:.2f}")
print(f"  R²:   {r2_score(y_test, y_pred_stack):.4f}")

# === Guardar modelo de stacking completo ===
os.makedirs("models", exist_ok=True)
joblib.dump(stack, "models/modelo_clientes.pkl")
print("\n✅ Modelo completo guardado como 'models/modelo_clientes.pkl'")
