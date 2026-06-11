# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 05:17:46 2026

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
plt.title("Spectres ÉCHANTILLON sur 24 heures")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- 1) Création DateHeure ---
df_spectres["DateHeure"] = df_spectres.apply(
    lambda r: pd.to_datetime(str(r["Date"]) + " " + str(r["Heure"])),
    axis=1
)

# --- 2) Filtrer spectres ÉCHANTILLON ---
ech = df_spectres[df_spectres["Type"] == "echantillon"]

# --- 3) Sélectionner les 24h ---
start = ech["DateHeure"].min()
end = start + pd.Timedelta(hours=24)
ech_24h = ech[(ech["DateHeure"] >= start) & (ech["DateHeure"] <= end)]

# --- 4) Construction matrice PCA ---
# Pivot : lignes = DateHeure, colonnes = Lambda(nm)
M = ech_24h.pivot_table(
    index="DateHeure",
    columns="Lambda(nm)",
    values="Absorbance"
)

# --- IMPORTANT : retirer colonnes avec valeurs manquantes ---
M = M.dropna(axis=1)

# --- 5) Appliquer PCA ---
pca = PCA(n_components=2)
scores = pca.fit_transform(M)

# --- 6) Récupération des résultats PCA ---
PCA1 = scores[:, 0]
PCA2 = scores[:, 1]

# --- 7) Affichage scatter PCA ---
plt.figure(figsize=(10,6))
plt.scatter(PCA1, PCA2, c=np.arange(len(PCA1)), cmap="viridis")
plt.colorbar(label="Ordre temporel (du plus ancien au plus récent)")

for i, date in enumerate(M.index):
    plt.text(PCA1[i], PCA2[i], date.strftime("%H:%M"), fontsize=8)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA des spectres ÉCHANTILLON (24 heures)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 8) Variance expliquée ---
print("Variance expliquée PC1 :", pca.explained_variance_ratio_[0])
print("Variance expliquée PC2 :", pca.explained_variance_ratio_[1])


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


from sklearn.cluster import KMeans

# --- K-means sur PC1 & PC2 ---
k = 3  # Nombre de clusters, tu peux tester 2, 3, 4...
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(scores)

# --- Affichage PCA avec clusters ---
plt.figure(figsize=(10,6))

scatter = plt.scatter(PCA1, PCA2, c=clusters, cmap="tab10", s=85)

# Ajout des heures sur les points
for i, date in enumerate(M.index):
    plt.text(PCA1[i], PCA2[i], date.strftime("%H:%M"), fontsize=8)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"PCA des spectres ÉCHANTILLON (24h) + Clustering K-means (k={k})")
plt.grid(True)

# Ajouter légende des clusters
plt.legend(*scatter.legend_elements(), title="Cluster")

plt.tight_layout()
plt.show()

from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --- UMAP sur les spectres (matrice M) ---
umap_model = UMAP(
    n_neighbors=10,       # taille du voisinage
    min_dist=0.1,         # distance minimale dans l’espace réduit
    n_components=2,       # projection en 2D
    random_state=0
)

umap_coords = umap_model.fit_transform(M)

U1 = umap_coords[:,0]
U2 = umap_coords[:,1]

# --- Clustering K-means sur UMAP ---
k = 3   # même nombre qu'avec PCA (ou tester plusieurs)
kmeans_umap = KMeans(n_clusters=k, random_state=0)
clusters_umap = kmeans_umap.fit_predict(umap_coords)

# --- Affichage UMAP + clusters ---
plt.figure(figsize=(10,6))
scatter = plt.scatter(U1, U2, c=clusters_umap, cmap="tab10", s=85)

# Ajouter les heures
for i, date in enumerate(M.index):
    plt.text(U1[i], U2[i], date.strftime("%H:%M"), fontsize=8)

plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title(f"UMAP des spectres ÉCHANTILLON (24h) + K-means (k={k})")
plt.grid(True)

# Légende des clusters
plt.legend(*scatter.legend_elements(), title="Cluster")

plt.tight_layout()
plt.show()


# Fusion cluster + paramètres chimiques
# DataFrame des clusters
df_clusters = pd.DataFrame({
    "DateHeure": M.index,
    "Cluster": clusters_umap,   # ou clusters si PCA
    "UMAP1": U1,
    "UMAP2": U2
})

df_all = pd.merge(df_clusters, df_params, on="DateHeure", how="inner")

print(df_all.head())

print("Corrélation Cluster ↔ DCO :", df_all["Cluster"].corr(df_all["DCO"]))
print("Corrélation Cluster ↔ COT :", df_all["Cluster"].corr(df_all["COT"]))
print("Corrélation Cluster ↔ MES :", df_all["Cluster"].corr(df_all["MES"]))


stats = df_all.groupby("Cluster")[["DCO","COT","MES"]].mean()
print(stats)

#Scatter DCO vs Cluster
plt.figure(figsize=(8,4))
plt.scatter(df_all["Cluster"], df_all["DCO"], c=df_all["Cluster"], cmap="tab10", s=80)
plt.xlabel("Cluster")
plt.ylabel("DCO")
plt.title("Corrélation Cluster ↔ DCO")
plt.grid(True)
plt.show()


#Scatter MES vs Cluster
plt.figure(figsize=(8,4))
plt.scatter(df_all["Cluster"], df_all["MES"], c=df_all["Cluster"], cmap="tab10", s=80)
plt.xlabel("Cluster")
plt.ylabel("MES")
plt.title("Corrélation Cluster ↔ MES")
plt.grid(True)
plt.show()


#Scatter COT vs Cluster
plt.figure(figsize=(8,4))
plt.scatter(df_all["Cluster"], df_all["COT"], c=df_all["Cluster"], cmap="tab10", s=80)
plt.xlabel("Cluster")
plt.ylabel("COT")
plt.title("Corrélation Cluster ↔ COT")
plt.grid(True)
plt.show()




import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from umap import UMAP
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

# ----------------------------------------------------
# 1) MATRICE SPECTRALE
# ----------------------------------------------------
M = df_spectres[df_spectres["Type"]=="echantillon"].pivot_table(
    index="DateHeure", columns="Lambda(nm)", values="Absorbance"
).dropna(axis=1)

# ----------------------------------------------------
# 2) PCA
# ----------------------------------------------------
pca = PCA(n_components=2)
pca_scores = pca.fit_transform(M)
P1, P2 = pca_scores[:,0], pca_scores[:,1]

# ----------------------------------------------------
# 3) UMAP
# ----------------------------------------------------
um = UMAP(n_components=2, random_state=0)
umap_scores = um.fit_transform(M)
U1, U2 = umap_scores[:,0], umap_scores[:,1]

# ----------------------------------------------------
# 4) CLUSTERING K-means
# ----------------------------------------------------
k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(umap_scores)

# DataFrame clusterisé
df_clusters = pd.DataFrame({
    "DateHeure": M.index,
    "Cluster": clusters,
    "UMAP1": U1,
    "UMAP2": U2
})

# ----------------------------------------------------
# 5) FUSION AVEC LES PARAMÈTRES CHIMIQUES
# ----------------------------------------------------
df_all = pd.merge(df_clusters, df_params, on="DateHeure", how="inner")

# ----------------------------------------------------
# 6) CRÉATION D'UN DOSSIER TEMPORAIRE POUR LES IMAGES
# ----------------------------------------------------
if not os.path.exists("figures"):
    os.makedirs("figures")

# ----------------------------------------------------
# 7) FIGURE PCA
# ----------------------------------------------------
plt.figure(figsize=(8,5))
plt.scatter(P1, P2, c=clusters, cmap="tab10")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Clustering")
pca_img = "figures/pca.png"
plt.savefig(pca_img, dpi=150)
plt.close()

# ----------------------------------------------------
# 8) FIGURE UMAP
# ----------------------------------------------------
plt.figure(figsize=(8,5))
plt.scatter(U1, U2, c=clusters, cmap="tab10")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("UMAP - Clustering")
umap_img = "figures/umap.png"
plt.savefig(umap_img, dpi=150)
plt.close()

# ----------------------------------------------------
# 9) FIGURES CHIMIE VS CLUSTER
# ----------------------------------------------------
for param in ["DCO","COT","MES"]:
    plt.figure(figsize=(8,5))
    plt.scatter(df_all["Cluster"], df_all[param], c=df_all["Cluster"], cmap="tab10")
    plt.xlabel("Cluster")
    plt.ylabel(param)
    plt.title(f"Cluster vs {param}")
    chem_img = f"figures/{param}.png"
    plt.savefig(chem_img, dpi=150)
    plt.close()
