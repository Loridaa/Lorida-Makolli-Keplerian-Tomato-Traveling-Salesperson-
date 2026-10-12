
from pathlib import Path


file_path = Path(__file__).resolve().parent.parent / "data" / "easy.kttsp"

domatet = []


with open(file_path, "r") as file:

    for rreshti in file:

        rreshti = rreshti.strip()

        if not rreshti or rreshti.startswith("c"):
            continue

       
        if rreshti.startswith("p"):
            parametrat = rreshti.split()[2:]
            continue

       
        orbita = list(map(float, rreshti.split()))
        domatet.append(orbita)

print("Numri i domateve:", len(domatet))
print("Parametrat e misionit:", parametrat)

print("Domatja e pare:", domatet[0])
print("Domatja e dyte:", domatet[1])
