import csv
import os

archivo = "dataset_rehabilitacion.csv"

print("\nRegistro de paciente\n")

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

print("\nDatos de la sesión")

rango = input("Rango de movimiento (%): ")
velocidad = input("Velocidad media (segundos): ")
errores = input("Número de errores: ")
repeticiones = input("Número de repeticiones: ")
sesiones = input("Sesiones estimadas: ")

existe = os.path.isfile(archivo)

with open(archivo,"a",newline="") as f:

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
    rango,
    velocidad,
    errores,
    repeticiones,
    sesiones
    ])

print("\nPaciente guardado correctamente")
