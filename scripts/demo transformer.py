# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:45:42 2026

@author: Administrateur
"""




# Import des bibliothèques nécessaires
import pandas as pd          # pour manipuler les tableaux de données
import numpy as np           # pour gérer les valeurs NaN
import glob                  # pour trouver tous les fichiers dans un dossier
import os                    # pour gérer les chemins
import matplotlib.pyplot as plt   # pour tracer les graphiques



# Chemin du dossier contenant les fichiers de spectres
folder = r"C:\Users\Administrateur\OneDrive\Desktop\visualisation Fp spectro"


# Recherche de tous les fichiers présents dans le dossier
files = glob.glob(os.path.join(folder, "*"))


# Liste qui va contenir tous les spectres
all_rows = []


# -------------------------
# LECTURE DES FICHIERS
# -------------------------


for file in files:   # boucle sur chaque fichier du dossier
    
    # ouverture du fichier
    with open(file, "r") as f:
        lines = f.readlines()   # lecture de toutes les lignes du fichier

    # ligne 1 du fichier = longueurs d'onde
    header = lines[1].split()

    # on récupère les longueurs d'onde (on ignore Date/Time et Status)
    wavelengths = [float(x) for x in header[2:]]

    # lecture des lignes contenant les spectres
    for line in lines[2:]:
        
        parts = line.split()   # séparation des valeurs de la ligne
        
        # sécurité si la ligne est vide
        if len(parts) < 5:
            continue
        
        # récupération des informations
        date = parts[0]        # date de mesure
        time = parts[1]        # heure de mesure
        status = parts[2]      # état de la mesure (Failure / OK)

        # valeurs du spectre
        values = parts[3:]

        # conversion en nombres (NaN si valeur manquante)
        values = [np.nan if v == "NaN" else float(v) for v in values]

        # création d'une ligne complète
        row = [date, time, status] + values

        # ajout à la liste globale
        all_rows.append(row)


# -------------------------
# CREATION DU TABLEAU FINAL
# -------------------------

# noms des colonnes
columns = ["date", "time", "status"] + wavelengths

# création du DataFrame pandas
df = pd.DataFrame(all_rows, columns=columns)

# affichage des dimensions du tableau
print(df.shape)

# affichage des premières lignes
print(df.head())


# -------------------------
# SAUVEGARDE DU TABLEAU
# -------------------------

# sauvegarde de toutes les données dans un fichier CSV
df.to_csv("spectres_combined.csv", index=False)


# -------------------------
# VISUALISATION D'UN SPECTRE
# -------------------------

# sélection du premier spectre du tableau
spectre = df.iloc[0, 3:]

# création du graphique
plt.figure()

# tracé intensité vs longueur d'onde
plt.plot(wavelengths, spectre)

# étiquettes du graphique
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("absorbance")

# titre
plt.title("Spectre d'absorbance mesuré")

# affichage
plt.show()


#### Etape 1 selection des colonnes spectrales

# -------------------------
# SELECTION DES SPECTRES UV-VIS
# -------------------------

# Colonnes spectrales uniquement (en ignorant date, time, status)
spectral_cols = df.columns[3:]

spectres = df[spectral_cols]

# Suppression des longueurs d'onde entièrement NaN
#spectres = spectres.dropna(axis=1)


# Interpolation des NaN le long des longueurs d’onde
spectres = spectres.interpolate(axis=1)



# Suppression des lignes contenant encore des NaN
spectres = spectres.dropna()

print("Dimensions des spectres :", spectres.shape)



## creation de la PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
spectres_scaled = scaler.fit_transform(spectres)


from sklearn.decomposition import PCA

pca = PCA(n_components=1)

# Projection des spectres
indicator_pca = pca.fit_transform(spectres_scaled).flatten()

print("Variance expliquée :", pca.explained_variance_ratio_[0])



## creation de la serie temporelles

# -------------------------
# SERIE TEMPORELLE
# -------------------------

indicator_series = pd.Series(indicator_pca)

plt.figure(figsize=(10,4))
plt.plot(indicator_series.values[:1000])
plt.xlabel("Temps")
plt.ylabel("1ère composante PCA")
plt.title("Evolution temporelle issue des spectres UV-Vis (PCA)")
plt.show()



## creation de sequences temporelles

def make_sequences(data, window=24):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X, y = make_sequences(indicator_series.values, window=24)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# MODELE TRANSFORMER POUR LA PREDICTION TEMPORELLE (DEMO)
# ============================================================

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) VISUALISATION D'UNE ENTREE ET DE LA SORTIE A PREDIRE
# ------------------------------------------------------------

# Exemple de fenêtre temporelle
exemple_X = X[0]    # 24 valeurs PCA
exemple_y = y[0]    # valeur suivante

print("Entrée du modèle (24 pas de temps PCA) :")
print(exemple_X)

print("\nValeur réelle à prédire :")
print(exemple_y)

# ------------------------------------------------------------
# 2) CONVERSION DES DONNEES EN TENSEURS PYTORCH
# ------------------------------------------------------------

# X : (N, 24)  -> (N, 24, 1)
X_torch = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y_torch = torch.tensor(y, dtype=torch.float32)

print("\nDimensions PyTorch X :", X_torch.shape)
print("Dimensions PyTorch y :", y_torch.shape)



# ============================================================
# MODELE TRANSFORMER - VERSION DEMO STABLE (AVEC BATCHS)
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) PREPARATION DES DONNEES POUR LE DEEP LEARNING
# ------------------------------------------------------------

X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)

batch_size = 256   # petit batch = mémoire maîtrisée
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------------------------------------------------------------
# 2) MODELE TRANSFORMER SIMPLIFIE ET ROBUSTE
# ------------------------------------------------------------

class TimeTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Linear(1, 8)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=8,
            nhead=2,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )

        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x[:, -1, :]          # dernier pas de temps
        return self.fc(x)

model = TimeTransformer()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------------------------------------------------
# 3) ENTRAINEMENT DU MODELE (PAR MINI-BATCHS)
# ------------------------------------------------------------

n_epochs = 5
losses = []

for epoch in range(n_epochs):
    epoch_loss = 0.0

    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb).squeeze()
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    losses.append(epoch_loss / len(loader))
    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {losses[-1]:.4f}")

# ------------------------------------------------------------
# 4) COURBE D'APPRENTISSAGE
# ------------------------------------------------------------

plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Apprentissage du Transformer")
plt.show()

# ------------------------------------------------------------
# 5) PREDICTIONS (MODE EVALUATION)
# ------------------------------------------------------------

model.eval()
with torch.no_grad():
    y_pred = []
    for xb, _ in loader:
        y_pred.append(model(xb).squeeze())

y_pred = torch.cat(y_pred).numpy()

# ------------------------------------------------------------
# 6) GRAPHIQUE FINAL (DEMO)
# ------------------------------------------------------------

plt.figure(figsize=(12,5))
plt.plot(y[:500], label="Données réelles", color="black")
plt.plot(y_pred[:500], label="Prédictions Transformer", color="red")
plt.xlabel("Temps")
plt.ylabel("1ère composante PCA")
plt.title("Transformer - Prédiction temporelle UV-Vis")
plt.legend()
plt.show()


# ============================================================
# MODELE LSTM POUR LA PREDICTION TEMPORELLE (DEMO)
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) PREPARATION DES DONNEES (MEMES X, y QUE LE TRANSFORMER)
# ------------------------------------------------------------

X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)

batch_size = 256
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------------------------------------------------------------
# 2) DEFINITION DU MODELE LSTM
# ------------------------------------------------------------

class LSTMModel(nn.Module):
    """
    LSTM classique pour la prédiction temporelle
    """
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,     # variable PCA
            hidden_size=32,   # taille de la mémoire
            batch_first=True
        )

        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # x : (batch, temps, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]     # dernier pas de temps
        return self.fc(out)

model_lstm = LSTMModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)

# ------------------------------------------------------------
# 3) ENTRAINEMENT DU LSTM
# ------------------------------------------------------------

n_epochs = 5
losses_lstm = []

for epoch in range(n_epochs):
    epoch_loss = 0.0

    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model_lstm(xb).squeeze()
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    losses_lstm.append(epoch_loss / len(loader))
    print(f"LSTM Epoch {epoch+1}/{n_epochs} - Loss: {losses_lstm[-1]:.4f}")

# ------------------------------------------------------------
# 4) COURBE D'APPRENTISSAGE LSTM
# ------------------------------------------------------------

plt.figure()
plt.plot(losses_lstm)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Apprentissage du LSTM")
plt.show()

# ------------------------------------------------------------
# 5) PREDICTIONS DU LSTM
# ------------------------------------------------------------

model_lstm.eval()
with torch.no_grad():
    y_pred_lstm = []
    for xb, _ in loader:
        y_pred_lstm.append(model_lstm(xb).squeeze())

y_pred_lstm = torch.cat(y_pred_lstm).numpy()

# ------------------------------------------------------------
# 6) GRAPHIQUE LSTM : DONNEES REELLES VS PREDICTIONS
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))
plt.plot(y[:500], label="Données réelles", color="black")
plt.plot(y_pred_lstm[:500], label="Prédictions LSTM", color="blue")
plt.xlabel("Temps")
plt.ylabel("1ère composante PCA")
plt.title("LSTM - Prédiction temporelle UV-Vis")
plt.legend()
plt.show()




# ============================================================
# MODELE RANDOM FOREST (BASELINE NON TEMPORELLE)
# ============================================================

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) FORMATAGE DES DONNEES POUR RANDOM FOREST
# ------------------------------------------------------------

# Random Forest attend une matrice 2D : (samples, features)
# Ici : 24 valeurs PCA par sample
X_rf = X          # shape (N, 24)
y_rf = y          # shape (N,)

print("X_rf shape:", X_rf.shape)
print("y_rf shape:", y_rf.shape)

# ------------------------------------------------------------
# 2) DEFINITION DU MODELE RANDOM FOREST
# ------------------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# ------------------------------------------------------------
# 3) ENTRAINEMENT DU MODELE
# ------------------------------------------------------------

rf_model.fit(X_rf, y_rf)

print("Random Forest entraîné.")

# ------------------------------------------------------------
# 4) PREDICTIONS
# ------------------------------------------------------------

y_pred_rf = rf_model.predict(X_rf)

# ------------------------------------------------------------
# 5) GRAPHIQUE : RANDOM FOREST VS DONNEES REELLES
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))
plt.plot(y_rf[:500], label="Données réelles", color="black")
plt.plot(y_pred_rf[:500], label="Random Forest", color="green")
plt.xlabel("Temps")
plt.ylabel("1ère composante PCA")
plt.title("Random Forest - Prédiction temporelle UV-Vis")
plt.legend()
plt.show()



# ============================================================
# GRAPHIQUE COMPARATIF FINAL : RF vs LSTM vs TRANSFORMER
# ============================================================

n_display = 500

plt.figure(figsize=(14, 6))

plt.plot(
    y[:n_display],
    label="Données réelles (PCA)",
    color="black",
    linewidth=2
)

plt.plot(
    y_pred_rf[:n_display],
    label="Random Forest",
    color="green",
    alpha=0.8
)

plt.plot(
    y_pred_lstm[:n_display],
    label="LSTM",
    color="blue",
    alpha=0.8
)

plt.plot(
    y_pred[:n_display],
    label="Transformer",
    color="red",
    alpha=0.6
)

plt.xlabel("Temps")
plt.ylabel("1ère composante PCA")
plt.title("Comparaison des modèles pour la prédiction temporelle UV-Vis")
plt.legend()
plt.grid(True)
plt.show()




























