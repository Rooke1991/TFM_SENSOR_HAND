import pandas as pd
import joblib

# ==========================
# CARGAR MODELO
# ==========================

modelo = joblib.load("modelo_prediccion_sesiones.pkl")

# ==========================
# CARGAR DATASET
# ==========================

data = pd.read_csv("dataset_pacientes/grupo_1/dataset_rehabilitacion.csv")

print("\nPacientes disponibles:\n")

for p in data["paciente"]:
    print("-",p)

# ==========================
# PEDIR PACIENTE
# ==========================

nombre = input("\n¿De qué paciente quieres la predicción?: ")

paciente = data[data["paciente"] == nombre]

if paciente.empty:

    print("\nPaciente no encontrado")
    exit()

# ==========================
# PREPARAR DATOS
# ==========================

lesion_map = {
"tendon":0,
"fractura":1,
"neurologica":2
}

edad = paciente["edad"].values[0]
lesion = lesion_map[paciente["lesion"].values[0]]
rango = paciente["rango_movimiento"].values[0]
velocidad = paciente["velocidad"].values[0]
repeticiones = paciente["repeticiones"].values[0]

X = [[edad,lesion,rango,velocidad,repeticiones]]

# ==========================
# PREDICCIÓN
# ==========================

prediccion = modelo.predict(X)

print("\nPaciente:",nombre)
print("Sesiones estimadas de rehabilitación:",round(prediccion[0]))
