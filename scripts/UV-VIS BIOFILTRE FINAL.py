
# %% 
#import

# %%
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
# Force UTF-8 encoding for stdout on Windows to prevent UnicodeEncodeError
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
# %%
# ------------------------------------------------------
# Dossier contenant les fichiers TXT
# ------------------------------------------------------
# On cherche le dossier localement dans le repo en priorité, avec repli vers le chemin d'origine
script_dir = os.path.dirname(os.path.abspath(__file__))
local_folder = os.path.abspath(os.path.join(script_dir, "..", "data", "fichier EF biofiltres"))
if os.path.exists(local_folder):
    folder = local_folder
else:
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
# 1) Construction DateHeure dans les spectres
# =======================================================
ech = df_spectres[df_spectres["Type"] == "echantillon"].copy()

ech["DateHeure"] = pd.to_datetime(
    ech["Date"].astype(str) + " " + ech["Heure"].astype(str)
)

# =======================================================
# 2) Matrice X (spectres)
# =======================================================
X = ech.pivot_table(
    index="DateHeure",
    columns="Lambda(nm)",
    values="Absorbance"
).sort_index()

# =======================================================
# 3) Matrice Y (paramètres chimiques)
# =======================================================
df_params["DateHeure"] = pd.to_datetime(df_params["DateHeure"])

Y = df_params.set_index("DateHeure").sort_index()

# =======================================================
# 4) ALIGNEMENT PROPRE (TRÈS IMPORTANT)
# =======================================================
X, Y = X.align(Y, join="inner", axis=0)

print("Shape X:", X.shape)
print("Shape Y:", Y.shape)





# SECTION PLS 


from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# =======================================================
# 1) MATRICE X : spectres UV-Vis
# =======================================================

ech = df_spectres[df_spectres["Type"] == "echantillon"]

X = ech.pivot_table(
    index=["Date", "Heure"],
    columns="Lambda(nm)",
    values="Absorbance"
)

X = X.sort_index()

# =======================================================
# 2) MATRICE Y : paramètres chimiques
# =======================================================

Y = df_params[["DCO", "N-NO2-NO3", "COT", "MES"]]

# =======================================================
# 3) Normalisation
# =======================================================

scalerX = StandardScaler()
X_scaled = scalerX.fit_transform(X)

scalerY = StandardScaler()
Y_scaled = scalerY.fit_transform(Y)

# =======================================================
# 4) Recherche du meilleur nombre de composantes PLS
# =======================================================

max_comp = min(10, X.shape[0]-1)

r2_cv = []

for ncomp in range(1, max_comp + 1):

    pls_test = PLSRegression(n_components=ncomp)

    Y_cv_scaled = cross_val_predict(
        pls_test,
        X_scaled,
        Y_scaled,
        cv=5
    )

    r2 = r2_score(Y_scaled, Y_cv_scaled)
    r2_cv.append(r2)

best_ncomp = np.argmax(r2_cv) + 1

print("\n=== OPTIMISATION PLS ===")
print(f"Nombre optimal de composantes = {best_ncomp}")
print(f"R² CV max = {max(r2_cv):.3f}")

# =======================================================
# 5) Construction du modèle final
# =======================================================

pls = PLSRegression(n_components=best_ncomp)
pls.fit(X_scaled, Y_scaled)

# =======================================================
# 6) Validation croisée réaliste
# =======================================================

Y_pred_scaled = cross_val_predict(
    pls,
    X_scaled,
    Y_scaled,
    cv=5
)

Y_pred = scalerY.inverse_transform(Y_pred_scaled)

# =======================================================
# 7) DataFrame des prédictions
# =======================================================

df_pred = pd.DataFrame(
    Y_pred,
    columns=["DCO_pred", "NO3_pred", "COT_pred", "MES_pred"]
)

# =======================================================
# 8) Fonction d'évaluation robuste
# =======================================================

def evaluer_modele(y_true, y_pred, nom):

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    if rmse != 0:
        rpd = np.std(y_true) / rmse
    else:
        rpd = np.nan

    print(f"\n=== {nom} ===")
    print(f"R²   = {r2:.3f}")
    print(f"RMSE = {rmse:.3f}")
    print(f"MAE  = {mae:.3f}")
    print(f"RPD  = {rpd:.3f}")

# =======================================================
# 9) Évaluation paramètre par paramètre
# =======================================================

evaluer_modele(Y["DCO"], df_pred["DCO_pred"], "DCO")
evaluer_modele(Y["N-NO2-NO3"], df_pred["NO3_pred"], "NO2-NO3")
evaluer_modele(Y["COT"], df_pred["COT_pred"], "COT")
evaluer_modele(Y["MES"], df_pred["MES_pred"], "MES")

# =======================================================
# 10) Coefficients spectraux du modèle PLS
# =======================================================

coef_pls = pd.DataFrame(
    pls.coef_.T,
    index=X.columns.astype(float),
    columns=["DCO", "NO3", "COT", "MES"]
)

print("\n=== COEFFICIENTS PLS ===")
print(coef_pls.head())

# =======================================================
# 11) Signature spectrale des coefficients
# =======================================================

plt.figure(figsize=(14,8))

for col in coef_pls.columns:
    plt.plot(coef_pls.index, coef_pls[col], label=col)

plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Coefficient PLS")
plt.title("Signature spectrale du modèle PLS")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# =======================================================
# 12) Réel vs prédit
# =======================================================

params = [
    ("DCO", "DCO_pred"),
    ("N-NO2-NO3", "NO3_pred"),
    ("COT", "COT_pred"),
    ("MES", "MES_pred")
]

for vrai, pred in params:

    plt.figure(figsize=(6,6))

    plt.scatter(Y[vrai], df_pred[pred])

    mini = min(Y[vrai].min(), df_pred[pred].min())
    maxi = max(Y[vrai].max(), df_pred[pred].max())

    plt.plot([mini, maxi], [mini, maxi])

    plt.xlabel("Valeurs réelles")
    plt.ylabel("Valeurs prédites")
    plt.title(f"PLS : {vrai}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# PCA

# =======================================================
# CONSTRUCTION PROPRE DES DONNÉES PCA / PLS
# =======================================================

# --- Spectres échantillon ---
ech = df_spectres[df_spectres["Type"] == "echantillon"].copy()

# Création DateHeure
ech["DateHeure"] = pd.to_datetime(
    ech["Date"].astype(str) + " " + ech["Heure"].astype(str)
)

# =======================================================
# MATRICE X
# =======================================================
X = ech.pivot_table(
    index="DateHeure",
    columns="Lambda(nm)",
    values="Absorbance"
)

# IMPORTANT
X = X.sort_index()

# =======================================================
# MATRICE Y
# =======================================================
df_params["DateHeure"] = pd.to_datetime(df_params["DateHeure"])

Y = df_params.copy()
Y = Y.set_index("DateHeure")
Y = Y.sort_index()

# =======================================================
# FUSION ROBUSTE (MIEUX QUE ALIGN)
# =======================================================
df_final = X.merge(
    Y,
    left_index=True,
    right_index=True,
    how="inner"
)

# =======================================================
# SÉPARATION X / Y
# =======================================================
X = df_final[X.columns]

Y = df_final[["DCO","N-NO2-NO3","COT","MES"]]

print(X.shape)
print(Y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42, shuffle=True
)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)



from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_s)
X_test_pca = pca.transform(X_test_s)


plt.figure()

plt.scatter(X_train_pca[:,0], X_train_pca[:,1], label="Train")
plt.scatter(X_test_pca[:,0], X_test_pca[:,1], label="Test")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA train/test séparation")
plt.legend()
plt.grid()
plt.show()


# Variance expliquée par chaque composante
explained_var = pca.explained_variance_ratio_

print("\n=== VARIANCE EXPLIQUÉE PCA ===")
for i, var in enumerate(explained_var):
    print(f"PC{i+1} : {var:.3f} ({var*100:.1f} %)")

# Variance cumulée
cum_var = np.cumsum(explained_var)

print("\nVariance cumulée :")
for i, var in enumerate(cum_var):
    print(f"PC1 à PC{i+1} : {var:.3f} ({var*100:.1f} %)")



plt.figure(figsize=(8,5))

plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')

plt.axhline(0.95, linestyle='--')  # seuil 95%
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance expliquée cumulée")
plt.title("Choix du nombre de composantes PCA")
plt.grid(True)

plt.show()

# loading PCA

plt.figure(figsize=(10,6))

plt.plot(X.columns.astype(float), pca.components_[0], label="PC1")
plt.plot(X.columns.astype(float), pca.components_[1], label="PC2")

plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Loading")
plt.title("Loadings PCA")
plt.legend()
plt.grid()

plt.show()


# K-Means


X_scaled_full = scaler.fit_transform(X)
X_pca_full = pca.fit_transform(X_scaled_full)

# choix du nombre optimal (methode du coude)
from sklearn.cluster import KMeans

inertia = []

K_range = range(1, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_pca_full)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Nombre de clusters k")
plt.ylabel("Inertia")
plt.title("Méthode du coude")
plt.grid()
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0)

clusters = kmeans.fit_predict(X_pca_full)

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca_full[:,0],
    X_pca_full[:,1],
    c=clusters,
    cmap='viridis'
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Clustering K-means sur PCA")
plt.colorbar(label="Cluster")
plt.grid()

plt.show()

df_clusters = X.copy()
df_clusters["cluster"] = clusters


spectres_moyens = df_clusters.groupby("cluster").mean()

plt.figure(figsize=(10,6))

for c in spectres_moyens.index:
    plt.plot(
        spectres_moyens.columns.astype(float),
        spectres_moyens.loc[c],
        label=f"Cluster {c}"
    )

plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Absorbance")
plt.title("Fingerprint spectral par cluster")
plt.legend()
plt.grid()

plt.show()


df_final_cluster = df_final.copy()
df_final_cluster["cluster"] = clusters

print(df_final_cluster.groupby("cluster")[["DCO","N-NO2-NO3","COT","MES"]].mean())




# ==========================================================
# SUITE LOGIQUE APRÈS LE K-MEANS
# Fingerprinting + UMAP + Corrélations chimiques
# ==========================================================

from umap import UMAP
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

# ==========================================================
# 1) AJOUT DES CLUSTERS AU DATAFRAME FINAL
# ==========================================================

df_final_cluster = df_final.copy()

df_final_cluster["Cluster"] = clusters

print(df_final_cluster.head())

# ==========================================================
# 2) PCA AVEC HEURES SUR LES POINTS
# ==========================================================

plt.figure(figsize=(12,7))

scatter = plt.scatter(
    X_pca_full[:,0],
    X_pca_full[:,1],
    c=clusters,
    cmap="tab10",
    s=85
)

# affichage des heures
for i, date in enumerate(df_final_cluster.index):

    plt.text(
        X_pca_full[i,0],
        X_pca_full[i,1],
        date.strftime("%H:%M"),
        fontsize=8
    )

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("PCA + KMeans + Heures")

plt.grid(True)

plt.legend(*scatter.legend_elements(), title="Cluster")

plt.show()

# ==========================================================
# 3) UMAP SUR LES SPECTRES UV-VIS
# ==========================================================

umap_model = UMAP(
    n_neighbors=10,
    min_dist=0.05,
    n_components=2,
    random_state=42
)

X_umap = umap_model.fit_transform(X_scaled_full)

U1 = X_umap[:,0]
U2 = X_umap[:,1]

# ==========================================================
# 4) KMEANS SUR UMAP
# ==========================================================

kmeans_umap = KMeans(
    n_clusters=3,
    random_state=42
)

clusters_umap = kmeans_umap.fit_predict(X_umap)

# ==========================================================
# 5) UMAP + CLUSTERS + HEURES
# ==========================================================

plt.figure(figsize=(12,7))

scatter = plt.scatter(
    U1,
    U2,
    c=clusters_umap,
    cmap="tab10",
    s=85
)

# ajout des heures
for i, date in enumerate(df_final_cluster.index):

    plt.text(
        U1[i],
        U2[i],
        date.strftime("%H:%M"),
        fontsize=8
    )

plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")

plt.title("UMAP + KMeans + Heures")

plt.grid(True)

plt.legend(*scatter.legend_elements(), title="Cluster")

plt.show()

# ==========================================================
# 6) CORRÉLATIONS CLUSTERS ↔ PARAMÈTRES CHIMIQUES
# ==========================================================

print("\n=== CORRÉLATIONS CLUSTERS / PARAMÈTRES ===\n")

for param in ["DCO", "N-NO2-NO3", "COT", "MES"]:

    corr = df_final_cluster["Cluster"].corr(
        df_final_cluster[param]
    )

    print(f"{param} : r = {corr:.3f}")

# ==========================================================
# 7) MOYENNES CHIMIQUES PAR CLUSTER
# ==========================================================

stats_cluster = df_final_cluster.groupby("Cluster")[[
    "DCO",
    "N-NO2-NO3",
    "COT",
    "MES"
]].mean()

print("\n=== MOYENNES CHIMIQUES PAR CLUSTER ===\n")

print(stats_cluster)

# ==========================================================
# 8) SCATTERS PARAMÈTRES CHIMIQUES VS CLUSTER
# ==========================================================

for param in ["DCO", "N-NO2-NO3", "COT", "MES"]:

    plt.figure(figsize=(8,5))

    plt.scatter(
        df_final_cluster["Cluster"],
        df_final_cluster[param],
        c=df_final_cluster["Cluster"],
        cmap="tab10",
        s=85
    )

    plt.xlabel("Cluster")
    plt.ylabel(param)

    plt.title(f"Cluster ↔ {param}")

    plt.grid(True)

    plt.show()

# ==========================================================
# 9) FINGERPRINT SPECTRAL MOYEN PAR CLUSTER
# ==========================================================

df_fp = X.copy()

df_fp["Cluster"] = clusters

spectres_moyens = df_fp.groupby("Cluster").mean()

plt.figure(figsize=(12,7))

for c in spectres_moyens.index:

    plt.plot(
        spectres_moyens.columns.astype(float),
        spectres_moyens.loc[c],
        linewidth=2,
        label=f"Cluster {c}"
    )

plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Absorbance")

plt.title("Fingerprint spectral moyen par cluster")

plt.legend()
plt.grid(True)

plt.show()

# ==========================================================
# 10) FINGERPRINT AVEC VARIABILITÉ
# ==========================================================

plt.figure(figsize=(12,7))

wavelengths = X.columns.astype(float)

for c in np.unique(clusters):

    subset = X[clusters == c]

    mean_spec = subset.mean(axis=0)

    std_spec = subset.std(axis=0)

    plt.plot(
        wavelengths,
        mean_spec,
        linewidth=2,
        label=f"Cluster {c}"
    )

    plt.fill_between(
        wavelengths,
        mean_spec - std_spec,
        mean_spec + std_spec,
        alpha=0.2
    )

plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Absorbance")

plt.title("Fingerprint spectral + variabilité")

plt.legend()
plt.grid(True)

plt.show()

# ==========================================================
# 11) CORRÉLATION SPECTRALE ↔ DCO
# ==========================================================

correlations_dco = []

for col in X.columns:

    corr = np.corrcoef(
        X[col],
        Y["DCO"]
    )[0,1]

    correlations_dco.append(corr)

plt.figure(figsize=(12,6))

plt.plot(
    X.columns.astype(float),
    correlations_dco
)

plt.axhline(0, color='black')

plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Corrélation")

plt.title("Corrélation spectrale ↔ DCO")

plt.grid(True)

plt.show()

# ==========================================================
# 12) CORRÉLATION SPECTRALE ↔ COT
# ==========================================================

correlations_cot = []

for col in X.columns:

    corr = np.corrcoef(
        X[col],
        Y["COT"]
    )[0,1]

    correlations_cot.append(corr)

plt.figure(figsize=(12,6))

plt.plot(
    X.columns.astype(float),
    correlations_cot
)

plt.axhline(0, color='black')

plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Corrélation")

plt.title("Corrélation spectrale ↔ COT")

plt.grid(True)

plt.show()

# ==========================================================
# 13) HEATMAP SPECTRALE
# ==========================================================

X_heat = X.copy()

X_heat["Cluster"] = clusters

X_heat = X_heat.sort_values("Cluster")

clusters_sorted = X_heat["Cluster"]

X_heat = X_heat.drop(columns="Cluster")

plt.figure(figsize=(16,8))

sns.heatmap(
    X_heat,
    cmap="viridis",
    cbar_kws={"label":"Absorbance"}
)

plt.title("Heatmap spectrale UV-VIS")

plt.xlabel("Longueur d'onde")

plt.ylabel("Échantillons")

plt.show()

# ==========================================================
# 14) DENDROGRAMME HIÉRARCHIQUE
# ==========================================================

linked = linkage(
    X_scaled_full,
    method='ward'
)

plt.figure(figsize=(14,6))

dendrogram(linked)

plt.title("Dendrogramme spectral")

plt.xlabel("Échantillons")
plt.ylabel("Distance")

plt.grid(True)

plt.show()

# ==========================================================
# 15) ÉVOLUTION TEMPORELLE DES CLUSTERS
# ==========================================================

plt.figure(figsize=(14,5))

plt.plot(
    df_final_cluster.index,
    df_final_cluster["Cluster"],
    marker='o'
)

plt.xlabel("Temps")
plt.ylabel("Cluster")

plt.title("Évolution temporelle des clusters")

plt.grid(True)

plt.show()

# ==========================================================
# 16) TABLEAU FINAL RÉSUMÉ
# ==========================================================

resume_clusters = df_final_cluster.groupby("Cluster")[[
    "DCO",
    "N-NO2-NO3",
    "COT",
    "MES"
]].agg(["mean", "std", "min", "max"])

print("\n=== RÉSUMÉ COMPLET DES CLUSTERS ===\n")

print(resume_clusters)

















