import serial
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, Counter
import os
import csv
import time

# =========================
# CONFIG
# =========================

PUERTO = "/dev/ttyACM0"
BAUDIOS = 9600
VENTANA = 5
OBJETIVO_REP = 6

BASE_DIR = "dataset_pacientes"
HISTORIAL_FILE = "historial_pacientes.csv"

# =========================
# VOICE
# =========================

def hablar(txt):
    os.system(f'espeak "{txt}"')

# =========================
# PATIENT REGISTRATION
# =========================

print("\nIntelligent Rehabilitation System\n")

nombre = input("Patient name: ")
edad = input("Age: ")

print("\nInjury type")
print("1 Tendon")
print("2 Fracture")
print("3 Neurological")

op = input("Option: ")

lesiones = {
"1":"tendon",
"2":"fractura",
"3":"neurologica"
}

lesion = lesiones.get(op,"unknown")

print("\nFinger to rehabilitate")
print("1 indice")
print("2 corazon")
print("3 anular")
print("4 pulgar")

op = input("Option: ")

dedos = {
"1":"indice",
"2":"corazon",
"3":"anular",
"4":"pulgar"
}

dedo_objetivo = dedos.get(op)

# =========================
# WELCOME
# =========================

hablar(f"Welcome {nombre}")
hablar(f"Please start moving your finger {dedo_objetivo}")

# =========================
# LOAD MODEL
# =========================

modelo = joblib.load("modelo_gestos.pkl")

# =========================
# SERIAL
# =========================

ser = serial.Serial(PUERTO, BAUDIOS, timeout=1)

# =========================
# BUFFERS
# =========================

pulgar = deque(maxlen=VENTANA)
indice = deque(maxlen=VENTANA)
corazon = deque(maxlen=VENTANA)
anular = deque(maxlen=VENTANA)
menique = deque(maxlen=VENTANA)

historial = deque(maxlen=5)
ultimo_gesto = None

repeticiones = 0

valores = []

# =========================
# GRAPH
# =========================

plt.ion()
fig, ax = plt.subplots()

linea, = ax.plot([], [])

ax.set_ylim(0,1023)

# =========================
# MAIN LOOP
# =========================

try:

    while True:

        line = ser.readline().decode("utf-8",errors="ignore").strip()

        if not line or not line[0].isdigit():
            continue

        data=line.split(",")

        if len(data)==6:

            ts,p,i,c,a,m = map(int,data)

            pulgar.append(p)
            indice.append(i)
            corazon.append(c)
            anular.append(a)
            menique.append(m)

            if len(pulgar) < VENTANA:
                continue

            entrada = pd.DataFrame([[

                np.mean(pulgar),
                np.mean(indice),
                np.mean(corazon),
                np.mean(anular),
                np.mean(menique)

            ]],columns=[
                "pulgar",
                "indice",
                "corazon",
                "anular",
                "menique"
            ])

            gesto = modelo.predict(entrada)[0]

            historial.append(gesto)

            gesto_estable = Counter(historial).most_common(1)[0][0]

            if gesto_estable != ultimo_gesto:

                print("GESTO:", gesto_estable)

                if gesto_estable == dedo_objetivo:

                    repeticiones += 1

                    hablar("Good job")

                    print("Repetition", repeticiones, "/", OBJETIVO_REP)

                ultimo_gesto = gesto_estable

            # =====================
            # GRAPH ONLY ACTIVE FINGER
            # =====================

            if dedo_objetivo == "indice":
                valores.append(i)

            elif dedo_objetivo == "corazon":
                valores.append(c)

            elif dedo_objetivo == "anular":
                valores.append(a)

            elif dedo_objetivo == "pulgar":
                valores.append(p)

            linea.set_data(range(len(valores)), valores)

            ax.set_xlim(0, len(valores))

            ax.set_title(f"{dedo_objetivo.upper()} MOVING")

            plt.pause(0.001)

            if repeticiones >= OBJETIVO_REP:

                hablar("Session complete")

                break

except KeyboardInterrupt:

    pass

ser.close()

# =========================
# METRICS
# =========================

rango = max(valores) - min(valores)

rango_porcentaje = round((rango/1023)*100,2)

velocidad = round(len(valores)/repeticiones,2)

# =========================
# SAVE BIG DATA DATASET
# =========================

os.makedirs(BASE_DIR,exist_ok=True)

total_pacientes = 0

for root,dirs,files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith(".csv"):
            total_pacientes += 1

grupo = (total_pacientes // 20) + 1

grupo_dir = f"{BASE_DIR}/grupo_{grupo}"

os.makedirs(grupo_dir,exist_ok=True)

csv_path = f"{grupo_dir}/dataset_rehabilitacion.csv"

existe = os.path.isfile(csv_path)

with open(csv_path,"a",newline="") as f:

    writer = csv.writer(f)

    if not existe:

        writer.writerow([
        "paciente",
        "edad",
        "lesion",
        "dedo",
        "rango_movimiento",
        "velocidad",
        "repeticiones"
        ])

    writer.writerow([
        nombre,
        edad,
        lesion,
        dedo_objetivo,
        rango_porcentaje,
        velocidad,
        repeticiones
    ])

print("\nData saved in", grupo_dir)

# =========================
# UPDATE SESSION HISTORY
# =========================

if not os.path.exists(HISTORIAL_FILE):

    df = pd.DataFrame(columns=[
        "paciente",
        "edad",
        "sesiones_predichas",
        "sesiones_realizadas"
    ])

    df.to_csv(HISTORIAL_FILE,index=False)

hist = pd.read_csv(HISTORIAL_FILE)

registro = hist[hist["paciente"] == nombre]

if registro.empty:

    nuevo = pd.DataFrame([[

        nombre,
        edad,
        sesiones_predichas,
        1

    ]],columns=[
        "paciente",
        "edad",
        "sesiones_predichas",
        "sesiones_realizadas"
    ])

    hist = pd.concat([hist,nuevo])

else:

    idx = registro.index[0]

    hist.loc[idx,"sesiones_realizadas"] += 1

hist.to_csv(HISTORIAL_FILE,index=False)

print("\nSession recorded in history")

hablar("The registration is saved")
