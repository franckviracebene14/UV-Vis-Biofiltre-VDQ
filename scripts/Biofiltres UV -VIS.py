# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:04:04 2026

@author: Administrateur
"""

# -*- coding: utf-8 -*-
"""
Traitement UV-VIS Biofiltre - Version finale corrigée
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# ------------------------------------------------------
# Dossier contenant les fichiers TXT
# ------------------------------------------------------
folder = r"C:\Users\Administrateur\OneDrive\Desktop\fichier EF biofiltres"

# ------------------------------------------------------
# Types de spectres présents dans les fichiers
# ------------------------------------------------------
TYPES = {
    "Spectre échantillon": "echantillon",
    "Spectre restitué": "restitue",
    "Spectre différence": "difference"
}

# ------------------------------------------------------
# Conversion nom fichier → datetime
# ------------------------------------------------------
def extraire_datetime(nom):
    base = nom.replace("Result__", "").replace(".txt", "")
    d, h = base.split("_")
    return dt.datetime(
        int(d[:4]), int(d[4:6]), int(d[6:8]),
        int(h[:2]), int(h[2:4])
    )

# ------------------------------------------------------
# 1) EXTRACTION DES SPECTRES
# ------------------------------------------------------
spectres_list = []

for fichier in sorted(os.listdir(folder)):
    if not fichier.endswith(".txt"):
        continue

    path = os.path.join(folder, fichier)
    with open(path, "r", encoding="latin-1") as f:
        lignes = [l.strip() for l in f.readlines()]

    dtfile = extraire_datetime(fichier)
    type_actuel = None
    data_temp = []

    for ligne in lignes:

        # Détection d'un nouveau spectre
        if ligne in TYPES:
            if type_actuel and data_temp:
                df = pd.DataFrame(data_temp, columns=["Lambda(nm)", "Absorbance"])
                df["Type"] = type_actuel
                df["Date"] = dtfile.date()
                df["Heure"] = dtfile.time()
                df["DateHeure"] = dtfile
                df["Fichier"] = fichier
                spectres_list.append(df)

            type_actuel = TYPES[ligne]
            data_temp = []
            continue

        # Lecture d'une ligne "lambda;abs"
        if ";" in ligne:
            try:
                lam, ab = ligne.split(";")
                data_temp.append([float(lam), float(ab)])
            except:
                pass

    # Dernier spectre du fichier
    if type_actuel and data_temp:
        df = pd.DataFrame(data_temp, columns=["Lambda(nm)", "Absorbance"])
        df["Type"] = type_actuel
        df["Date"] = dtfile.date()
        df["Heure"] = dtfile.time()
        df["DateHeure"] = dtfile
        df["Fichier"] = fichier
        spectres_list.append(df)

df_spectres = pd.concat(spectres_list, ignore_index=True)

# ------------------------------------------------------
# 2) EXTRACTION DES PARAMÈTRES CHIMIQUES
# ------------------------------------------------------
params_list = []

for fichier in sorted(os.listdir(folder)):
    if not fichier.endswith(".txt"):
        continue

    path = os.path.join(folder, fichier)
    with open(path, "r", encoding="latin-1") as f:
        lignes = [l.strip() for l in f.readlines()]

    dtfile = extraire_datetime(fichier)

    for i, ligne in enumerate(lignes):
        if ligne.startswith("DCO;N-NO2-NO3;COT;MES"):
            v = lignes[i+1].split(";")
            params_list.append({
                "DateHeure": dtfile,
                "DCO": float(v[0]),
                "N-NO2-NO3": float(v[1]),
                "COT": float(v[2]),
                "MES": float(v[3])
            })

df_params = pd.DataFrame(params_list).sort_values("DateHeure")

# ------------------------------------------------------
# 3) FUSION SPECTRES + PARAMÈTRES
# ------------------------------------------------------
df_merged = pd.merge(
    df_spectres,
    df_params,
    on="DateHeure",
    how="left"
)

# ------------------------------------------------------
# 4) CORRÉLATIONS ABSORBANCE(λ) ↔ PARAMÈTRES
# ------------------------------------------------------
ech = df_merged[df_merged["Type"] == "echantillon"]

lambdas = sorted(ech["Lambda(nm)"].unique())

correlations = {
    "DCO": [],
    "N-NO2-NO3": [],
    "COT": [],
    "MES": []
}

for lam in lambdas:
    sub = ech[ech["Lambda(nm)"] == lam]
    for param in correlations.keys():
        if sub[param].nunique() > 1:
            correlations[param].append(sub["Absorbance"].corr(sub[param]))
        else:
            correlations[param].append(np.nan)

# ------------------------------------------------------
# 5) TRACÉ DES CORRÉLATIONS
# ------------------------------------------------------
plt.figure(figsize=(12,7))
for param, values in correlations.items():
    plt.plot(lambdas, values, label=param)

plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Corrélation Pearson")
plt.title("Corrélations : Absorbance(λ) ↔ paramètres chimiques")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()