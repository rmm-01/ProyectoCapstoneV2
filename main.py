import subprocess
import sys

MODULOS_DISPONIBLES = {
    "demanda_clientes": "src/pronostico_demanda_clientes",
    "demanda_materiales": "src/pronostico_demanda_materiales",
    "exceso_stock": "src/pronostico_exceso_stock",
    "quiebre_stock": "src/pronostico_quiebre_stock",
    "riesgo_clientes": "src/pronostico_riesgo_clientes",
    "stock_optimo": "src/pronostico_stock_optimo"
}

def ejecutar_modulo(modulo):
    if modulo not in MODULOS_DISPONIBLES:
        print(f"❌ Módulo '{modulo}' no es válido. Usa uno de: {list(MODULOS_DISPONIBLES.keys())}")
        return

    ruta = MODULOS_DISPONIBLES[modulo]
    
    print(f"\n🚀 Ejecutando entrenamiento del módulo '{modulo}'...\n")
    subprocess.run(["python", f"{ruta}/train_model.py"], check=True)

    print(f"\n📊 Ejecutando evaluación del módulo '{modulo}'...\n")
    subprocess.run(["python", f"{ruta}/evaluate.py"], check=True)

    print(f"\n✅ Proceso completo para el módulo '{modulo}'.\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("❗ Uso: python main.py <modulo>")
        print(f"   Módulos disponibles: {list(MODULOS_DISPONIBLES.keys())}")
    else:
        ejecutar_modulo(sys.argv[1])
