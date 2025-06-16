import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import cargar_datos, procesar_datos, preparar_features
from sklearn.pipeline import Pipeline

# Carga y procesamiento
file = "data/datos_limpios.xlsx"
df_demanda, df_stock = cargar_datos(file)
df = procesar_datos(df_demanda, df_stock)
X, y, preprocessor = preparar_features(df)

# Split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# Modelos base
estimators = [
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("gb", GradientBoostingRegressor(random_state=42)),
    ("knn", KNeighborsRegressor())
]
meta_model = LinearRegression()

# Métricas individuales
print("\n🔹 MÉTRICAS INDIVIDUALES DE MODELOS BASE:")
for name, model in estimators:
    model.fit(X_train, y_train)
    y_pred_base = model.predict(X_test)
    mse_base = mean_squared_error(y_test, y_pred_base)
    rmse_base = np.sqrt(mse_base)
    r2_base = r2_score(y_test, y_pred_base)
    print(f"\nModelo base: {name.upper()}")
    print(f"  MSE:  {mse_base:.4f}")
    print(f"  RMSE: {rmse_base:.4f}")
    print(f"  R²:   {r2_base:.4f}")

# Stacking
stack_model = StackingRegressor(estimators=estimators, final_estimator=meta_model, cv=5)
stack_model.fit(X_train, y_train)


#  Métricas stacking
y_pred_stack = stack_model.predict(X_test)
mse_stack = mean_squared_error(y_test, y_pred_stack)
rmse_stack = np.sqrt(mse_stack)
r2_stack = r2_score(y_test, y_pred_stack)

print("\n🔸 MÉTRICAS DEL MODELO STACKING (FINAL):")
print(f"  MSE:  {mse_stack:.4f}")
print(f"  RMSE: {rmse_stack:.4f}")
print(f"  R²:   {r2_stack:.4f}")
# Guardar
os.makedirs("models", exist_ok=True)
modelo_final = Pipeline([
    ("preprocessor", preprocessor),
    ("model", stack_model)
])
joblib.dump(modelo_final, "models/modelo_exceso.pkl")
print("\n✅ Modelo guardado como 'models/modelo_exceso.pkl'")
