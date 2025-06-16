import os
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import cargar_datos, procesar_datos, preparar_features

print("\n🚀 Ejecutando entrenamiento del modulo 'stock_optimo'...")

demanda, stock = cargar_datos("data/datos_limpios.xlsx")
df = procesar_datos(demanda, stock)
X, y = preparar_features(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
et = ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=42)

stack = StackingRegressor(
    estimators=[("rf", rf), ("gb", gb), ("et", et)],
    final_estimator=RidgeCV(),
    cv=5
)

stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n🔸 MÉTRICAS DEL MODELO STACKING (FINAL):")
print(f"MAE:  {mae:.2f}")

print(f"R²:   {r2:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(stack, "models/modelo_stock_optimo.pkl")
print("\n✅ Modelo guardado como 'models/modelo_stock_optimo.pkl'")