import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# ==========================
# CARGAR DATASET
# ==========================

data = pd.read_csv("dataset_pacientes/grupo_1/dataset_rehabilitacion.csv")

print("\nDatos cargados:")
print(data.head())

# ==========================
# CONVERTIR LESION
# ==========================

lesion_map = {
"tendon":0,
"fractura":1,
"neurologica":2
}

data["lesion"] = data["lesion"].map(lesion_map)

# ==========================
# GENERAR SESIONES CLINICAS
# ==========================

def calcular_sesiones(row):

    rango = row["rango_movimiento"]
    velocidad = row["velocidad"]
    edad = row["edad"]
    lesion = row["lesion"]

    sesiones = 10

    # rango movimiento
    if rango < 20:
        sesiones += 12
    elif rango < 40:
        sesiones += 8
    elif rango < 60:
        sesiones += 5
    else:
        sesiones += 2

    # velocidad movimiento
    if velocidad > 60:
        sesiones += 4
    elif velocidad > 40:
        sesiones += 2

    # edad
    if edad > 60:
        sesiones += 4
    elif edad > 40:
        sesiones += 2

    # lesión
    if lesion == 2:  # neurologica
        sesiones += 6
    elif lesion == 1: # fractura
        sesiones += 3

    return sesiones

data["sesiones"] = data.apply(calcular_sesiones, axis=1)

# ==========================
# VARIABLES
# ==========================

X = data[[
"edad",
"lesion",
"rango_movimiento",
"velocidad",
"repeticiones"
]]

y = data["sesiones"]

# ==========================
# ENTRENAR MODELO
# ==========================

modelo = RandomForestRegressor(n_estimators=200)

modelo.fit(X,y)

joblib.dump(modelo,"modelo_prediccion_sesiones.pkl")

print("\nModelo Big Data entrenado correctamente")
