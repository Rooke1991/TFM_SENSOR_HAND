import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# ==========================
# CARGAR MODELO
# ==========================

modelo = joblib.load("modelo_prediccion_sesiones.pkl")

# ==========================
# DATASET PACIENTES
# ==========================

data = pd.read_csv("dataset_pacientes/grupo_1/dataset_rehabilitacion.csv")

print("\nPacientes disponibles:\n")

for p in data["paciente"]:
    print("-",p)

# ==========================
# SELECCIONAR PACIENTE
# ==========================

nombre = input("\n¿De qué paciente quieres ver la rehabilitación?: ")

paciente = data[data["paciente"] == nombre]

if paciente.empty:
    print("Paciente no encontrado")
    exit()

edad = paciente["edad"].values[0]
lesion = paciente["lesion"].values[0]
rango = paciente["rango_movimiento"].values[0]
velocidad = paciente["velocidad"].values[0]
repeticiones = paciente["repeticiones"].values[0]

# ==========================
# MAP LESION
# ==========================

lesion_map = {
"tendon":0,
"fractura":1,
"neurologica":2
}

lesion_num = lesion_map[lesion]

X = [[edad,lesion_num,rango,velocidad,repeticiones]]

# ==========================
# PREDICCION
# ==========================

sesiones_predichas = round(modelo.predict(X)[0])

# ==========================
# HISTORIAL PACIENTES
# ==========================

historial_file = "historial_pacientes.csv"

if not os.path.exists(historial_file):

    df_hist = pd.DataFrame(columns=[
        "paciente",
        "edad",
        "sesiones_predichas",
        "sesiones_realizadas"
    ])

    df_hist.to_csv(historial_file,index=False)

hist = pd.read_csv(historial_file)

registro = hist[hist["paciente"] == nombre]

# ==========================
# PACIENTE NUEVO
# ==========================

if registro.empty:

    sesiones_realizadas = 0

    nuevo = pd.DataFrame([[

        nombre,
        edad,
        sesiones_predichas,
        sesiones_realizadas

    ]],columns=[
        "paciente",
        "edad",
        "sesiones_predichas",
        "sesiones_realizadas"
    ])

    hist = pd.concat([hist,nuevo])

    hist.to_csv(historial_file,index=False)

else:

    sesiones_realizadas = registro["sesiones_realizadas"].values[0]
    sesiones_predichas = registro["sesiones_predichas"].values[0]

# ==========================
# SESIONES RESTANTES
# ==========================

sesiones_restantes = sesiones_predichas - sesiones_realizadas

# ==========================
# INTERFAZ VISUAL
# ==========================

fig, ax = plt.subplots(2,1,figsize=(8,6))

info = f"""
PACIENTE: {nombre}
Edad: {edad}
Lesión: {lesion}

Sesiones totales estimadas: {sesiones_predichas}
Sesiones realizadas: {sesiones_realizadas}

Sesiones restantes: {sesiones_restantes}
"""

ax[0].text(0.1,0.5,info,fontsize=14)

ax[0].axis("off")

# ==========================
# BARRA DE PROGRESO
# ==========================

progreso = sesiones_realizadas / sesiones_predichas

ax[1].barh(["Rehabilitación"],[progreso])

ax[1].set_xlim(0,1)

ax[1].set_title("Progreso de rehabilitación")

plt.tight_layout()

plt.show()
