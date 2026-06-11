# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:07:27 2026

@author: Administrateur
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import os
import pandas as pd
import datetime as dt
import numpy as np
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
# Fonction qui transforme le nom du fichier en Date/Heure
# Ex: Result__20250821_2343.txt → 2025-08-21 23:43
# ------------------------------------------------------
def extraire_datetime(nom):
    base = nom.replace("Result__", "").replace(".txt", "")
    d, h = base.split("_")
    return dt.datetime(
        int(d[:4]), int(d[4:6]), int(d[6:8]),
        int(h[:2]), int(h[2:4])
    )

# ------------------------------------------------------
# 1) EXTRACTION DES SPECTRES (3 par fichier)
# ------------------------------------------------------

spectres_list = []   # contiendra toutes les lignes de spectres

for fichier in sorted(os.listdir(folder)):

    if not fichier.endswith(".txt"):
        continue

    # Lire le fichier
    path = os.path.join(folder, fichier)
    with open(path, "r", encoding="latin-1") as f:
        lignes = [l.strip() for l in f.readlines()]

    # Extraire la date et l'heure
    dtfile = extraire_datetime(fichier)

    type_actuel = None        # type du spectre en cours
    data_temp = []            # liste temporaire Lambda/Abs

    for ligne in lignes:

        # Si la ligne annonce un spectre
        if ligne in TYPES:
            # Si un spectre précédent existe → on le sauvegarde en DataFrame
            if type_actuel and data_temp:
                df = pd.DataFrame(data_temp, columns=["Lambda(nm)", "Absorbance"])
                df["Type"] = type_actuel
                df["Date"] = dtfile.date()
                df["Heure"] = dtfile.time()
                df["Fichier"] = fichier
                spectres_list.append(df)

            # Début d’un nouveau spectre
            type_actuel = TYPES[ligne]
            data_temp = []
            continue

        # Lecture des données numériques Lambda;Abs
        if ";" in ligne:
            try:
                lam, ab = ligne.split(";")
                data_temp.append([float(lam), float(ab)])
            except:
                pass  # si ce n'est pas une valeur numérique, on ignore

    # Sauvegarde du dernier spectre rencontré
    if type_actuel and data_temp:
        df = pd.DataFrame(data_temp, columns=["Lambda(nm)", "Absorbance"])
        df["Type"] = type_actuel
        df["Date"] = dtfile.date()
        df["Heure"] = dtfile.time()
        df["Fichier"] = fichier
        spectres_list.append(df)

# Fusion de tous les spectres en un seul DataFrame
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

    # On recherche la ligne DCO;NO3;COT;MES
    for i, ligne in enumerate(lignes):
        if ligne.startswith("DCO;N-NO2-NO3;COT;MES"):

            valeurs = lignes[i+1].split(";")

            params_list.append({
                "DateHeure": dtfile,
                "DCO": float(valeurs[0]),
                "N-NO2-NO3": float(valeurs[1]),
                "COT": float(valeurs[2]),
                "MES": float(valeurs[3])
            })
            break

# DataFrame des paramètres chimiques
df_params = pd.DataFrame(params_list).sort_values("DateHeure").reset_index(drop=True)


# ------------------------------------------------------
# 3) AFFICHAGE CONSOLE
# ------------------------------------------------------

for fichier in df_spectres["Fichier"].unique():

    subset = df_spectres[df_spectres["Fichier"] == fichier]
    d = subset["Date"].iloc[0]
    h = subset["Heure"].iloc[0]

    
    print(f"=== {fichier}   ({d} {h}) ===")
    

    print("\n--- SPECTRE ÉCHANTILLON ---")
    print(subset[subset["Type"] == "echantillon"].head())

    print("\n--- SPECTRE RESTITUÉ ---")
    print(subset[subset["Type"] == "restitue"].head())

    print("\n--- SPECTRE DIFFÉRENCE ---")
    print(subset[subset["Type"] == "difference"].head())


print("DataFrame  pour Analyse Spectrale")
print(df_params)

import matplotlib.pyplot as plt
import pandas as pd

# --- 1) Création de la colonne DateHeure ---
df_spectres["DateHeure"] = df_spectres.apply(
    lambda r: pd.to_datetime(str(r["Date"]) + " " + str(r["Heure"])),
    axis=1
)

# --- 2) Filtrer uniquement les SPECTRES ÉCHANTILLON ---
ech = df_spectres[df_spectres["Type"] == "echantillon"]

# --- 3) Définir la fenêtre de 24 heures ---
start = ech["DateHeure"].min()
end = start + pd.Timedelta(hours=24)

ech_24h = ech[(ech["DateHeure"] >= start) & (ech["DateHeure"] <= end)]

# --- 4) Tracé avec légende = Date + Heure ---
plt.figure(figsize=(14, 7))

for dh in sorted(ech_24h["DateHeure"].unique()):
    sub = ech_24h[ech_24h["DateHeure"] == dh]
    legend_txt = dh.strftime("%Y-%m-%d %H:%M")
    plt.plot(sub["Lambda(nm)"], sub["Absorbance"], label=legend_txt)

plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Absorbance")
plt.title("Spectres ÉCHANTILLON biofiltre sur 24 heures")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================================
# 1) Sélection des spectres ÉCHANTILLON
# ==========================================================
ech = df_spectres[df_spectres["Type"] == "echantillon"]

# Pivot : une ligne = 1 heure, colonnes = Lambda(nm)
ech_matrix = ech.pivot_table(
    index=["Date", "Heure"],
    columns="Lambda(nm)",
    values="Absorbance"
)

# Assurer tri temporel identique à df_params
ech_matrix = ech_matrix.reset_index(drop=True)
dfp = df_params.reset_index(drop=True)

# ==========================================================
# 2) Calcul des corrélations pour chaque paramètre
# ==========================================================
lambdas = ech_matrix.columns.values.astype(float)

correlations = {
    "DCO": [],
    "N-NO2-NO3": [],
    "COT": [],
    "MES": []
}

# Pour chaque longueur d'onde λ
for lam in lambdas:
    A = ech_matrix[lam].values  # absorbance sur 24 heures

    correlations["DCO"].append(np.corrcoef(A, dfp["DCO"])[0,1])
    correlations["N-NO2-NO3"].append(np.corrcoef(A, dfp["N-NO2-NO3"])[0,1])
    correlations["COT"].append(np.corrcoef(A, dfp["COT"])[0,1])
    correlations["MES"].append(np.corrcoef(A, dfp["MES"])[0,1])

# ==========================================================
# 3) Trouver la meilleure longueur d'onde pour chaque paramètre
# ==========================================================
meilleurs_lambda = {
    param: lambdas[np.argmax(np.abs(correlations[param]))]
    for param in correlations
}

print("\n=== MEILLEURES LONGUEURS D’ONDE PAR CORRÉLATION ===\n")
for param, lam in meilleurs_lambda.items():
    print(f"➡ Meilleure λ pour {param} = {lam:.1f} nm "
          f"(corr = {max(correlations[param], key=abs):.3f})")

# ==========================================================
# 4) Tracés des courbes de corrélation (optionnels)
# ==========================================================
plt.figure(figsize=(12,7))
plt.plot(lambdas, correlations["DCO"], label="DCO")
plt.plot(lambdas, correlations["N-NO2-NO3"], label="NO2-NO3")
plt.plot(lambdas, correlations["COT"], label="COT")
plt.plot(lambdas, correlations["MES"], label="MES")
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Corrélation r")
plt.title("Corrélation Absorbance(λ) ↔ Paramètres chimiques")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



zones = {
    "DCO":      (190, 250),
    "N-NO2-NO3":(200, 230),   # zone nitrate uniquement
    "COT":      (190, 250),
    "MES":      (300, 400)    # diffusion
}

print("\n=== MEILLEURES LONGUEURS D’ONDE PAR ZONE SPÉCIFIQUE ===\n")

for param, (lmin, lmax) in zones.items():

    mask = (lambdas >= lmin) & (lambdas <= lmax)
    lamb = lambdas[mask]
    corr = np.array(correlations[param])[mask]

    idx = np.argmax(np.abs(corr))
    print(f"{param}: λ = {lamb[idx]:.1f} nm (corr = {corr[idx]:.3f})")
    
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# 1) EXTRACTION DES SPECTRES ÉCHANTILLON (X) + PARAMÈTRES (Y)
# ==========================================================

# On garde uniquement les spectres échantillons

# ==========================================================
# FUSION SPECTRES + PARAMÈTRES CHIMIQUES
# (INDISPENSABLE pour utiliser df_merged)
# ==========================================================

df_spectres["DateHeure"] = df_spectres.apply(
    lambda r: pd.to_datetime(str(r["Date"]) + " " + str(r["Heure"])),
    axis=1
)

df_params["DateHeure"] = pd.to_datetime(df_params["DateHeure"])

df_merged = pd.merge(
    df_spectres,
    df_params,
    on="DateHeure",
    how="left"
)





ech = df_merged[df_merged["Type"] == "echantillon"]

# Matrice X : spectres UV-VIS (1 ligne = 1 spectre = 1 heure)
# Colonnes = lambda 190–800 nm
X = ech.pivot_table(
    index="DateHeure",
    columns="Lambda(nm)",
    values="Absorbance"
).sort_index()

# Matrice Y : paramètres chimiques alignés sur X
Y = df_params.set_index("DateHeure").loc[X.index][["DCO", "COT", "MES", "N-NO2-NO3"]]

# Passage en numpy
X_mat = X.values
Y_mat = Y.values

# ==========================================================
# 2) MODÈLE PLS MULTIVARIÉE (PLS2)
# ==========================================================

# Le fabricant utilise généralement entre 5 et 10 composantes latentes
pls = PLSRegression(n_components=7)

# Prédiction par validation croisée (K-fold = 5)
Y_pred = cross_val_predict(pls, X_mat, Y_mat, cv=5)

# ==========================================================
# 3) CALCUL DES INDICATEURS DE QUALITÉ
# ==========================================================

param_names = ["DCO", "COT", "MES", "NO2-NO3"]

def NSE(obs, pred):
    """Nash–Sutcliffe Efficiency (très utilisé en modélisation hydrologique)."""
    return 1 - np.sum((pred - obs)**2) / np.sum((obs - np.mean(obs))**2)

qualite = {}

for i,p in enumerate(param_names):
    obs = Y_mat[:,i]
    pred = Y_pred[:,i]

    r2 = r2_score(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    mae = mean_absolute_error(obs, pred)
    mape = np.mean(np.abs((obs - pred) / obs)) * 100
    nse = NSE(obs, pred)
    
    qualite[p] = {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "NSE": nse
    }

# Affichage console
print("\n========== Indicateurs de qualité PLS ==========\n")
for p in param_names:
    print(f"\n--- {p} ---")
    for k,v in qualite[p].items():
        print(f"{k} = {v:.4f}")

# ==========================================================
# 4) GRAPHIQUE : Mesuré vs Prédit pour chaque paramètre
# ==========================================================

plt.figure(figsize=(12,10))
for i,p in enumerate(param_names):
    obs = Y_mat[:,i]
    pred = Y_pred[:,i]
    
    plt.subplot(2,2,i+1)
    plt.scatter(obs, pred, alpha=0.8)
    plt.plot([obs.min(), obs.max()], [obs.min(), obs.max()], "r--")
    plt.xlabel("Mesuré")
    plt.ylabel("Prédit")
    plt.title(f"PLS - {p}")
    plt.grid(True)

plt.tight_layout()
plt.show()