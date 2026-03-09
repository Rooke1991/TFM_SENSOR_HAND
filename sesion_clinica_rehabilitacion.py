import serial
import joblib
import pandas as pd
import numpy as np
import csv
import os
import time
from collections import deque, Counter

# ==============================
# CONFIGURACIÓN
# ==============================

PUERTO = '/dev/ttyACM0'
BAUDIOS = 9600
VENTANA = 5
DATASET = "dataset_rehabilitacion.csv"

# ==============================
# VOZ
# ==============================

def hablar(texto):
    os.system(f'espeak "{texto}"')

# ==============================
# REGISTRO PACIENTE
# ==============================

print("\nSistema de rehabilitación de mano\n")

nombre = input("Nombre del paciente: ")
edad = input("Edad: ")

print("\nTipo de lesión:")
print("1 Tendón")
print("2 Fractura")
print("3 Neurológica")

op = input("Opción: ")

lesiones = {
"1":"tendon",
"2":"fractura",
"3":"neurologica"
}

lesion = lesiones.get(op,"desconocida")

print("\nPaciente registrado\n")

# ==============================
# SELECCIÓN EJERCICIO
# ==============================

print("Seleccione ejercicio:")

print("1 indice")
print("2 corazon")
print("3 anular")
print("4 pulgar")

op = input("Opción: ")

ejercicios = {
"1":"indice",
"2":"corazon",
"3":"anular",
"4":"pulgar"
}

objetivo = ejercicios.get(op)

hablar(f"Ejercicio del dedo {objetivo}")

print("\nComenzando ejercicio...\n")

# ==============================
# CARGAR MODELO
# ==============================

modelo = joblib.load("modelo_gestos.pkl")

# ==============================
# SERIAL
# ==============================

ser = serial.Serial(PUERTO, BAUDIOS, timeout=1)

# ==============================
# BUFFERS
# ==============================

pulgar = deque(maxlen=VENTANA)
indice = deque(maxlen=VENTANA)
corazon = deque(maxlen=VENTANA)
anular = deque(maxlen=VENTANA)
menique = deque(maxlen=VENTANA)

historial = deque(maxlen=5)

ultimo_gesto = None

repeticiones = 0
errores = 0

inicio_sesion = time.time()

valores_objetivo = []

# ==============================
# LOOP REHABILITACIÓN
# ==============================

try:

    while True:

        line = ser.readline().decode('utf-8',errors='ignore').strip()

        if not line or not line[0].isdigit():
            continue

        data = line.split(',')

        if len(data) == 6:

            ts,p,i,c,a,m = map(int,data)

            p = min(p,880)

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

                print("GESTO:",gesto_estable)

                if gesto_estable == objetivo:

                    hablar("Movimiento correcto")

                    repeticiones += 1

                    print("Repeticiones:",repeticiones)

                    if objetivo == "indice":
                        valores_objetivo.append(np.mean(indice))
                    elif objetivo == "corazon":
                        valores_objetivo.append(np.mean(corazon))
                    elif objetivo == "anular":
                        valores_objetivo.append(np.mean(anular))
                    elif objetivo == "pulgar":
                        valores_objetivo.append(np.mean(pulgar))

                elif gesto_estable != "reposo":

                    errores += 1

                    hablar("Movimiento incorrecto")

                ultimo_gesto = gesto_estable

            # parar después de 15 repeticiones
            if repeticiones >= 15:
                break

except KeyboardInterrupt:
    pass

ser.close()

# ==============================
# CÁLCULO MÉTRICAS
# ==============================

duracion = time.time() - inicio_sesion

if repeticiones > 0:
    velocidad = duracion / repeticiones
else:
    velocidad = duracion

if valores_objetivo:
    rango = max(valores_objetivo) - min(valores_objetivo)
else:
    rango = 0

# normalizar rango a porcentaje aproximado
rango_mov = round((rango / 1023) * 100,2)

# estimación simple sesiones
if rango_mov < 30:
    sesiones = 25
elif rango_mov < 50:
    sesiones = 18
elif rango_mov < 70:
    sesiones = 12
else:
    sesiones = 8

# ==============================
# GUARDAR DATASET
# ==============================

existe = os.path.isfile(DATASET)

with open(DATASET,"a",newline="") as f:

    writer = csv.writer(f)

    if not existe:
        writer.writerow([
        "paciente","edad","lesion",
        "rango_movimiento",
        "velocidad",
        "errores",
        "repeticiones",
        "sesiones"
        ])

    writer.writerow([
        nombre,
        edad,
        lesion,
        rango_mov,
        round(velocidad,2),
        errores,
        repeticiones,
        sesiones
    ])

# ==============================
# RESULTADOS
# ==============================

print("\nSesión finalizada\n")

print("Repeticiones:",repeticiones)
print("Errores:",errores)
print("Velocidad media:",round(velocidad,2))
print("Rango movimiento:",rango_mov,"%")
print("Sesiones estimadas:",sesiones)

hablar("Sesion finalizada")
