




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

