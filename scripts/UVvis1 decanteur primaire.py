# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 02:07:08 2026

@author: Administrateur
"""


# -*- coding: utf-8 -*-
"""
Version SIMPLIFIÉE basé DataFrame :
- Lecture des 3 spectres par fichier
- Lecture des paramètres DCO/NO3/COT/MES
- Construction DataFrames simples
- Affichage propre en console
"""

import os
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------------------
# Dossier contenant les fichiers TXT
# ------------------------------------------------------
folder = r"C:\Users\Administrateur\OneDrive\Desktop\fichier EF Decanteur primaire"

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






# ------------------------------------------------------
# FILTRER UNIQUEMENT LES SPECTRES ÉCHANTILLON
# ------------------------------------------------------
echantillon = df_spectres[df_spectres["Type"] == "echantillon"]

# Extraire les heures triées (sans doublons)
heures_triees = echantillon[["Date", "Heure"]].drop_duplicates()
heures_triees = heures_triees.sort_values(["Date", "Heure"]).reset_index(drop=True)

# ------------------------------------------------------
# TRACE UNIQUE : 24 SPECTRES SUPERPOSÉS (UNE SEULE COULEUR)
# ------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.title("Spectre Échantillon – 24 heures")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Absorbance")
plt.grid(True, alpha=0.3)

# Une seule couleur pour toutes les courbes
couleur = "blue"

# Boucle sur les 24 heures
for _, row in heures_triees.iterrows():

    spec = echantillon[
        (echantillon["Date"] == row["Date"]) &
        (echantillon["Heure"] == row["Heure"])
    ]

    if not spec.empty:
        plt.plot(spec["Lambda(nm)"], spec["Absorbance"],
                 color=couleur, alpha=0.4)

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
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# =======================================================
# 1) Matrice X : spectres échantillon (24h × N lambda)
# =======================================================
ech = df_spectres[df_spectres["Type"] == "echantillon"]

X = ech.pivot_table(
    index=["Date", "Heure"],
    columns="Lambda(nm)",
    values="Absorbance"
)

X = X.reset_index(drop=True)
lambdas = X.columns.astype(float).values

# =======================================================
# 2) Matrice Y : paramètres (DCO, NO3, COT, MES)
# =======================================================
Y = df_params[["DCO", "N-NO2-NO3", "COT", "MES"]].values

# =======================================================
# 3) Normalisation indispensable
# =======================================================
scalerX = StandardScaler()
X_scaled = scalerX.fit_transform(X)

scalerY = StandardScaler()
Y_scaled = scalerY.fit_transform(Y)

# =======================================================
# 4) Construire le modèle PLS du fabricant
#    (souvent 3 à 5 composantes latentes)
# =======================================================
pls = PLSRegression(n_components=5)
pls.fit(X_scaled, Y_scaled)


# =======================================================
# 5) Coefficients du modèle PLS
# =======================================================


coef_pls = pd.DataFrame(
    pls.coef_,                            # forme = (4, 610)
    index=["coef_DCO", "coef_NO3", "coef_COT", "coef_MES"],  
    columns=lambdas                       # 610 colonnes = toutes les λ
)

print("\n=== COEFFICIENTS DU MODÈLE PLS (modèle fabricant reconstruit) ===\n")
print(coef_pls.head())


# =======================================================
# 6) Prédiction test (calcul des paramètres à partir des spectres)
# =======================================================
Y_pred_scaled = pls.predict(X_scaled)
Y_pred = scalerY.inverse_transform(Y_pred_scaled)

df_pred = pd.DataFrame(Y_pred, columns=["DCO_pred", "NO3_pred", "COT_pred", "MES_pred"])
print("\n=== PRÉDICTIONS DU MODÈLE PLS ===\n")
print(df_pred.head())


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import numpy as np

def evaluer_modele(y_true, y_pred, nom):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    rpd = np.std(y_true) / rmse

    print(f"\n=== QUALITÉ DU MODÈLE PLS : {nom} ===")
    print(f"R²   : {r2:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"RPD  : {rpd:.3f}")

# Évaluation PLS pour chaque paramètre
evaluer_modele(df_params["DCO"],  df_pred["DCO_pred"],  "DCO")
evaluer_modele(df_params["N-NO2-NO3"], df_pred["NO3_pred"], "NO₂–NO₃")
evaluer_modele(df_params["COT"],  df_pred["COT_pred"], "COT")
evaluer_modele(df_params["MES"],  df_pred["MES_pred"], "MES")



# ---------------------------------------------------------
# Visualisation des coefficients PLS (signature spectrale)
# ---------------------------------------------------------

plt.figure(figsize=(14, 8))

# Tracer chaque PARAMÈTRE sur la même figure
plt.plot(coef_pls.columns, coef_pls.loc["coef_DCO"], label="DCO",  linewidth=2)
plt.plot(coef_pls.columns, coef_pls.loc["coef_NO3"], label="NO₂–NO₃", linewidth=2)
plt.plot(coef_pls.columns, coef_pls.loc["coef_COT"], label="COT", linewidth=2)
plt.plot(coef_pls.columns, coef_pls.loc["coef_MES"], label="MES", linewidth=2)

plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Coefficient PLS")
plt.title("Coefficients spectro-chimiques du modèle PLS (Type fabricant)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
