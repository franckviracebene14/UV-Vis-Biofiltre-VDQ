# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 03:25:14 2026

@author: Administrateur
"""

import os
import pandas as pd

# ======================================================
# Chemin du dossier contenant tes fichiers spectres
# ======================================================
folder = r"C:\Users\Administrateur\OneDrive\Desktop\fichier EF Decanteur primaire"# <-- adapte si besoin

# Liste pour stocker tous les DataFrames
all_spectres = []

# Mots-clés pour chaque spectre
TYPES = {
    "Spectre échantillon": "echantillon",
    "Spectre restitué": "restitue",
    "Spectre différence": "difference"
}

# ======================================================
# Lecture de tous les fichiers du dossier
# ======================================================

            
            
for filename in os.listdir(folder):
    if filename.lower().endswith(".txt"):
        
        filepath = os.path.join(folder, filename)
        print(f"\nLecture du fichier : {filename}")

        # -------------------------
        # LECTURE SÉCURISÉE DU FICHIER
        # -------------------------
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="latin-1") as f:
                lines = f.readlines()

        # ------------------------------------
        # Maintenant lines EXISTE TOUJOURS ! ✔
        # ------------------------------------

        current_type = None
        lambda_vals = []
        absorb_vals = []

        for line in lines:
            line = line.strip()
                        

            # Détection du type de spectre
            if line in TYPES:
                # S'il y avait déjà un spectre en cours → on sauvegarde
                if current_type is not None and lambda_vals:
                    df = pd.DataFrame({
                        "Lambda (nm)": lambda_vals,
                        "Absorbance": absorb_vals,
                        "Type": current_type,
                        "Fichier": filename
                    })
                    all_spectres.append(df)

                # Préparation pour un nouveau spectre
                current_type = TYPES[line]
                lambda_vals = []
                absorb_vals = []
                continue

            # Ignorer ligne d'entêtes
            if line.startswith("Lambda") or line == "":
                continue

            # Lire lignes du type "190.000;1.335"
            if ";" in line and current_type is not None:
                L, A = line.split(";")
                lambda_vals.append(float(L))
                absorb_vals.append(float(A))

        # Sauvegarder le dernier spectre du fichier
        if current_type is not None and lambda_vals:
            df = pd.DataFrame({
                "Lambda (nm)": lambda_vals,
                "Absorbance": absorb_vals,
                "Type": current_type,
                "Fichier": filename
            })
            all_spectres.append(df)

# ======================================================
# Fusion finale de tous les fichiers
# ======================================================
combined = pd.concat(all_spectres, ignore_index=True)

# ======================================================
# Export Excel
# ======================================================
output_file = "spectres_triples.xlsx"
combined.to_excel(output_file, index=False)

print("\n====================================================")
print(f"Fichier Excel '{output_file}' créé avec succès !")
print("====================================================")

# ======================================================
# 3 DataFrames séparés par type de spectre
# ======================================================

df_echantillon = combined[combined["Type"] == "echantillon"].reset_index(drop=True)
df_restitue    = combined[combined["Type"] == "restitue"].reset_index(drop=True)
df_difference  = combined[combined["Type"] == "difference"].reset_index(drop=True)

# Affichage console (optionnel)
print("\n=== SPECTRE ÉCHANTILLON ===")
print(df_echantillon.head())

print("\n=== SPECTRE RESTITUÉ ===")
print(df_restitue.head())

print("\n=== SPECTRE DIFFÉRENCE ===")
print(df_difference.head())

import datetime as dt

# ======================================================
# 1) EXTRAIRE Date + Heure depuis le nom du fichier
# ======================================================

def extract_datetime_from_filename(filename):
    # Ex: "Result__20250821_2343.txt"
    base = filename.replace("Result__", "").replace(".txt", "")
    
    date_str, time_str = base.split("_")   # ["20250821", "2343"]
    
    year  = int(date_str[0:4])
    month = int(date_str[4:6])
    day   = int(date_str[6:8])
    
    hour   = int(time_str[0:2])
    minute = int(time_str[2:4])
    
    return dt.datetime(year, month, day, hour, minute)

# Ajouter la colonne datetime dans combined
combined["DateHeure"] = combined["Fichier"].apply(extract_datetime_from_filename)

# ======================================================
# 2) CLASSEMENT PAR DATE + HEURE
# ======================================================
combined = combined.sort_values("DateHeure").reset_index(drop=True)

print("\n=== Vérification : premiers timestamps triés ===")
print(combined[["Fichier", "DateHeure"]].head())

# ======================================================
# 3) Export Excel : classé par heure
# ======================================================
with pd.ExcelWriter("spectres_tries_par_heure.xlsx", engine="openpyxl") as writer:
    combined.to_excel(writer, index=False)
print(combined[["Fichier","Type","Lambda (nm)","DateHeure"]].head(20))
print(combined.drop_duplicates("Fichier")[["Fichier","DateHeure"]])