# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 04:27:28 2026

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


import matplotlib.pyplot as plt

# ------------------------------------------------------
# Liste des 3 types de spectres à tracer
# ------------------------------------------------------
types_a_tracer = ["echantillon", "restitue", "difference"]

# On récupère la liste triée des heures disponibles
heures_triees = df_spectres[["Date", "Heure"]].drop_duplicates()
heures_triees = heures_triees.sort_values(["Date", "Heure"]).reset_index(drop=True)

# ------------------------------------------------------
# Tracer un graphique par type de spectre
# ------------------------------------------------------
for t in types_a_tracer:

    plt.figure(figsize=(10, 6))
    plt.title(f"SPECTRES {t.upper()} SUR 24 HEURES")

    # Parcours des 24 batchs (fichiers)
    for _, row in heures_triees.iterrows():

        # Récupération du spectre correspondant à cette date-heure
        spectre_1h = df_spectres[
            (df_spectres["Type"] == t) &
            (df_spectres["Date"] == row["Date"]) &
            (df_spectres["Heure"] == row["Heure"])
        ]

        # Si un spectre existe pour cette heure, on le trace
        if not spectre_1h.empty:
            plt.plot(
                spectre_1h["Lambda(nm)"],
                spectre_1h["Absorbance"],
                alpha=0.5,
                label=f"{row['Date']} {row['Heure']}"
            )

    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Absorbance")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.show()
    
    
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1) Sélectionner la date/heure 23h43 (n’importe quel jour)
# ------------------------------------------------------

# On sélectionne l'heure
heure_visee = dt.time(23, 43)     # 23h43

# On filtre les 3 spectres pour cette heure
spec_ech = df_spectres[
    (df_spectres["Type"] == "echantillon") &
    (df_spectres["Heure"] == heure_visee)
]

spec_res = df_spectres[
    (df_spectres["Type"] == "restitue") &
    (df_spectres["Heure"] == heure_visee)
]

spec_diff = df_spectres[
    (df_spectres["Type"] == "difference") &
    (df_spectres["Heure"] == heure_visee)
]

# ------------------------------------------------------
# 2) Vérification (optionnelle)
# ------------------------------------------------------
if spec_ech.empty:
    print("⚠ Aucun spectre trouvé pour 23h43 ! Vérifie la date/heure dans ton DataFrame.")

# ------------------------------------------------------
# 3) TRACÉ DES 3 COURBES SUPERPOSÉES
# ------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.title("Spectre - Échantillon vs Restitué vs Différence (23h43)")

plt.plot(spec_ech["Lambda(nm)"], spec_ech["Absorbance"],
         label="Échantillon", color="blue")

plt.plot(spec_res["Lambda(nm)"], spec_res["Absorbance"],
         label="Restitué", color="green")

plt.plot(spec_diff["Lambda(nm)"], spec_diff["Absorbance"],
         label="Différence", color="red")

plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Absorbance")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# ------------------------------------------------------
# Récupération des dates disponibles dans les données
# ------------------------------------------------------
dates_disponibles = sorted(df_spectres["Date"].unique())

# ------------------------------------------------------
# Pour chaque date → tracer Échantillon / Restitué / Différence
# ------------------------------------------------------
for date_sel in dates_disponibles:

    # Filtrer les données de cette date
    df_jour = df_spectres[df_spectres["Date"] == date_sel]

    # Création de la figure
    plt.figure(figsize=(10, 6))
    plt.title(f"Spectres Échantillon / Restitué / Différence\nDate : {date_sel}")

    # =======================
    # 1. SPECTRE ÉCHANTILLON
    # =======================
    ech = df_jour[df_jour["Type"] == "echantillon"]
    plt.plot(ech["Lambda(nm)"], ech["Absorbance"],
             color="blue", alpha=0.5, label="Échantillon")

    # =======================
    # 2. SPECTRE RESTITUÉ
    # =======================
    res = df_jour[df_jour["Type"] == "restitue"]
    plt.plot(res["Lambda(nm)"], res["Absorbance"],
             color="green", alpha=0.5, label="Restitué")

    # =======================
    # 3. SPECTRE DIFFÉRENCE
    # =======================
    diff = df_jour[df_jour["Type"] == "difference"]
    plt.plot(diff["Lambda(nm)"], diff["Absorbance"],
             color="red", alpha=0.5, label="Différence")

    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Absorbance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

#### regression lineaire ####

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# ==========================================================
# 1) Construire la matrice X des spectres échantillons (24 × N_lambda)
# ==========================================================

# Filtrer uniquement les spectres ÉCHANTILLON
ech = df_spectres[df_spectres["Type"] == "echantillon"]

# Pivot : lignes = heures, colonnes = lambda
X = ech.pivot_table(
    index=["Date", "Heure"],
    columns="Lambda(nm)",
    values="Absorbance"
)

# Assurer le tri temporel identique à df_params
X = X.reset_index(drop=True)

# ==========================================================
# 2) Construire les vecteurs y des paramètres (24 × 1)
# ==========================================================

y_DCO  = df_params["DCO"].values
y_NO3  = df_params["N-NO2-NO3"].values
y_COT  = df_params["COT"].values
y_MES  = df_params["MES"].values

# ==========================================================
# 3) Normalisation des données (fortement recommandée)
# ==========================================================

scalerX = StandardScaler()
X_scaled = scalerX.fit_transform(X)

# ==========================================================
# 4) Fonction pratique pour créer un modèle linéaire complet
# ==========================================================

def modele_lineaire(X_scaled, y, nom_parametre, lambda_cols):
    model = LinearRegression()
    model.fit(X_scaled, y)

    print(f"\n=== MODÈLE LINÉAIRE POUR {nom_parametre} ===")
    print(f"Intercept : {model.intercept_}")
    print(f"Nombre de coefficients : {len(model.coef_)}")

    # Construire DataFrame des coefficients correctement
    df_coef = pd.DataFrame({
        "Lambda(nm)": lambda_cols.astype(float),
        "Coef": model.coef_
    })

    print(df_coef.head())
    return model, df_coef






# ==========================================================
# 5) Appliquer la régression pour chaque paramètre
# ==========================================================

lambda_cols = X.columns.values  # colonnes des longueurs d’onde

model_DCO, coef_DCO = modele_lineaire(X_scaled, y_DCO, "DCO", lambda_cols)
model_NO3, coef_NO3 = modele_lineaire(X_scaled, y_NO3, "N-NO2-NO3", lambda_cols)
model_COT, coef_COT = modele_lineaire(X_scaled, y_COT, "COT", lambda_cols)
model_MES, coef_MES = modele_lineaire(X_scaled, y_MES, "MES", lambda_cols)

### parametres de qualité



from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import numpy as np

def evaluer_modele(y_true, y_pred, nom):
    r2  = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    rpd = np.std(y_true) / rmse

    print(f"\n=== QUALITÉ DU MODÈLE ({nom}) ===")
    print(f"R²   : {r2:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"RPD  : {rpd:.3f}")

    return r2, rmse, mae, rpd

# Exemple d’évaluation après prédiction modèle OLS
y_pred_DCO  = model_DCO.predict(X_scaled)
evaluer_modele(y_DCO, y_pred_DCO, "DCO (OLS)")

# Idem pour NO3, COT, MES



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



