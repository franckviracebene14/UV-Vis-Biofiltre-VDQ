# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:37:11 2026

@author: Administrateur
"""


import json
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import requests
import json



# ---------------------------------------------------------
# Download AO data from URL and save it as soi.json
# ---------------------------------------------------------

url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table"
text = requests.get(url).text

lines = text.strip().split("\n")
data_dict = {}

for line in lines[1:]:
    parts = line.split()
    year = parts[0]
    values = parts[1:]
    
    for i, v in enumerate(values):
        month = str(i+1).zfill(2)
        key = year + month          # AAAAMM
        data_dict[key] = float(v)

json_data = {"data": data_dict}

with open('soi.json', 'w') as file:
    json.dump(json_data, file, indent=4)

# ---------------------------------------------------------
# Read data from JSON file (exactly comme ton script)
# ---------------------------------------------------------

with open('soi.json', 'r') as file:
    soi_dict = json.load(file)

data = soi_dict['data']

# extract and convert values into float
soi = np.zeros(len(data))
time = np.zeros(len(data))

count = 0
for key, value in data.items():
    soi[count] = float(value)
    time[count] = float(key[:4]) + (float(key[4:]) - 1) / 12
    count += 1

# ---------------------------------------------------------
# Plot the data (style identique à ton image)
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(time, soi, label='Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Artic oscillation')
plt.legend()
plt.show()


# ---------------------------------------------------------
# ACF de la série originale (25 premiers délais)
# ---------------------------------------------------------

# ---------------------------------------------------------
# Create autocorrelogram with 95% confidence interval
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))
sm.graphics.tsa.plot_acf(soi, lags=25, alpha=0.05, ax=ax)
plt.title('Autocorrelogram with 95% Confidence Interval')
plt.show()

# ---------------------------------------------------------
# Performance des modèles AR(k) pour kmax = 25
# ---------------------------------------------------------


kmax = 25

rmse_vals = []
mae_vals = []
bias_vals = []
nse_vals = []

y = soi  # série à utiliser

for k in range(1, kmax + 1):
    model = AutoReg(y, lags=k, old_names=False).fit()
    
    y_pred = model.predict(start=k, end=len(y)-1)
    y_true = y[k:]
    
    # Metrics
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae  = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_true - y_pred)
    
    # Nash-Sutcliffe Efficiency
    nse  = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    
    rmse_vals.append(rmse)
    mae_vals.append(mae)
    bias_vals.append(bias)
    nse_vals.append(nse)

# ---------------------------------------------------------
# Figure synthèse (4 axes)
# ---------------------------------------------------------

fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

k_vals = np.arange(1, kmax+1)

axs[0].plot(k_vals, rmse_vals, marker='o')
axs[0].set_ylabel("RMSE")
axs[0].grid(True, alpha=0.3)

axs[1].plot(k_vals, mae_vals, marker='o', color='orange')
axs[1].set_ylabel("MAE")
axs[1].grid(True, alpha=0.3)

axs[2].plot(k_vals, bias_vals, marker='o', color='green')
axs[2].set_ylabel("Bias")
axs[2].grid(True, alpha=0.3)

axs[3].plot(k_vals, nse_vals, marker='o', color='red')
axs[3].set_ylabel("NSE")
axs[3].set_xlabel("k (lag)")
axs[3].grid(True, alpha=0.3)

plt.suptitle("Performance des modèles AR(k) (kmax = 25)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# ---------------------------------------------------------
# Figures individuelles pour les modèles AR(k), k = 1..5
# ---------------------------------------------------------

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf

kmax = 5

y = soi  # série

for k in range(1, kmax + 1):

    # -------- 1. Ajuster le modèle AR(k)
    model = AutoReg(y, lags=k, old_names=False).fit()

    # prédictions à partir du lag k
    y_pred = model.predict(start=k, end=len(y)-1)

    # vérité terrain correspondante
    y_true = y[k:]

    # erreurs
    errors = y_true - y_pred

    # -------- 2. Créer figure avec 3 sous‑axes
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # (1) Série originale + simulée
    axs[0].plot(y, label='Original')
    axs[0].plot(np.arange(k, len(y)), y_pred, label=f'AR({k})')
    axs[0].set_title(f'Modèle AR({k}) – Série Originale et Simulée')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # (2) Histogramme des erreurs
    axs[1].hist(errors, bins=20, color='gray', edgecolor='black')
    axs[1].set_title(f'Histogramme des Erreurs – AR({k})')
    axs[1].set_xlabel("Erreur")
    axs[1].set_ylabel("Fréquence")
    axs[1].grid(True, alpha=0.3)

    # (3) ACF des erreurs (10 premiers délais)
    plot_acf(errors, lags=10, alpha=0.05, ax=axs[2])
    axs[2].set_title(f'ACF des Erreurs – AR({k}) (10 lags)')
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    
  # ---------------------------------------------------------
# Figures pour AR(k) pour k = 1..5 (basées sur RMSE)
# ---------------------------------------------------------

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf

kmax = 5
y = soi

for k in range(1, kmax + 1):

    # -------- 1. Ajustement du modèle AR(k)
    model = AutoReg(y, lags=k, old_names=False).fit()

    # prédictions (à partir du lag k)
    y_pred = model.predict(start=k, end=len(y)-1)
    y_true = y[k:]

    # -------- Calcul du RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    print(f"RMSE pour AR({k}) = {rmse:.4f}")

    # erreurs
    errors = y_true - y_pred

    # -------- 2. Figure à 3 axes pour AR(k)
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(f"Modèle AR({k}) – Analyse basée sur RMSE", fontsize=14)

    # (1) Série originale + simulée
    axs[0].plot(y, label='Original')
    axs[0].plot(np.arange(k, len(y)), y_pred, label=f'Simulée AR({k})')
    axs[0].set_title(f"Série Originale vs Simulée – RMSE = {rmse:.4f}")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # (2) Histogramme des erreurs
    axs[1].hist(errors, bins=20, color='gray', edgecolor='black')
    axs[1].set_title("Histogramme des Erreurs")
    axs[1].set_xlabel("Erreur")
    axs[1].set_ylabel("Fréquence")
    axs[1].grid(True, alpha=0.3)

    # (3) ACF des erreurs (10 lags)
    plot_acf(errors, lags=10, alpha=0.05, ax=axs[2])
    axs[2].set_title("ACF des Erreurs (10 premiers délais)")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()  
    
    
# ---------------------------------------------------------
# Figure détaillée pour le modèle AR(5) avec RMSE
# ---------------------------------------------------------

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf

k = 5
y = soi

# -------- 1. Ajustement du modèle AR(5)
model = AutoReg(y, lags=k, old_names=False).fit()

# Prédictions à partir du lag k
y_pred = model.predict(start=k, end=len(y)-1)
y_true = y[k:]

# -------- 2. Calcul du RMSE
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
print(f"RMSE du modèle AR(5) : {rmse:.4f}")

# Erreurs
errors = y_true - y_pred

# -------- 3. Figure avec 3 sous-axes
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
fig.suptitle(f"Modèle AR(5) – Analyse basée sur le RMSE = {rmse:.4f}", fontsize=14)

# (1) Série originale vs simulée AR(5)
axs[0].plot(y, label='Série originale')
axs[0].plot(np.arange(k, len(y)), y_pred, label='Série simulée AR(5)')
axs[0].set_title("Originale vs Simulée (AR5)")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# (2) Histogramme des erreurs
axs[1].hist(errors, bins=20, color='gray', edgecolor='black')
axs[1].set_title("Histogramme des erreurs du modèle AR(5)")
axs[1].set_xlabel("Erreur")
axs[1].set_ylabel("Fréquence")
axs[1].grid(True, alpha=0.3)

# (3) ACF des erreurs (10 premiers délais)
plot_acf(errors, lags=10, alpha=0.05, ax=axs[2])
axs[2].set_title("ACF des erreurs (10 premiers délais)")
axs[2].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
    
    
    # ---------------------------------------------------------
# Figure détaillée pour le modèle AR(5) avec MAE
# ---------------------------------------------------------

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf

k = 5
y = soi

# -------- 1. Ajustement du modèle AR(5)
model = AutoReg(y, lags=k, old_names=False).fit()

# Prédictions à partir du lag k
y_pred = model.predict(start=k, end=len(y)-1)
y_true = y[k:]

# -------- 2. Calcul du MAE
mae = np.mean(np.abs(y_true - y_pred))
print(f"MAE du modèle AR(5) : {mae:.4f}")

# Erreurs
errors = y_true - y_pred

# -------- 3. Figure à 3 sous-axes
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
fig.suptitle(f"Modèle AR(5) – Analyse basée sur le MAE = {mae:.4f}", fontsize=14)

# (1) Série originale vs simulée AR(5)
axs[0].plot(y, label='Série originale')
axs[0].plot(np.arange(k, len(y)), y_pred, label='Série simulée AR(5)')
axs[0].set_title("Série Originale vs Simulée (AR5)")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# (2) Histogramme des erreurs
axs[1].hist(errors, bins=20, color='gray', edgecolor='black')
axs[1].set_title("Histogramme des erreurs du modèle AR(5)")
axs[1].set_xlabel("Erreur")
axs[1].set_ylabel("Fréquence")
axs[1].grid(True, alpha=0.3)

# (3) ACF des erreurs (10 premiers délais)
plot_acf(errors, lags=10, alpha=0.05, ax=axs[2])
axs[2].set_title("ACF des erreurs (10 premiers délais)")
axs[2].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
    
    
 # ---------------------------------------------------------
# Figure détaillée pour le modèle AR(5) avec Biais
# ---------------------------------------------------------

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf

k = 5
y = soi

# -------- 1. Ajustement du modèle AR(5)
model = AutoReg(y, lags=k, old_names=False).fit()

# Prédictions à partir du lag k
y_pred = model.predict(start=k, end=len(y)-1)
y_true = y[k:]

# -------- 2. Calcul du biais
bias = np.mean(y_true - y_pred)
print(f"Biais du modèle AR(5) : {bias:.4f}")

# Erreurs
errors = y_true - y_pred

# -------- 3. Figure à 3 sous-axes
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
fig.suptitle(f"Modèle AR(5) – Analyse basée sur le Biais = {bias:.4f}", fontsize=14)

# (1) Série originale vs simulée AR(5)
axs[0].plot(y, label='Série originale')
axs[0].plot(np.arange(k, len(y)), y_pred, label='Série simulée AR(5)')
axs[0].set_title("Série Originale vs Simulée (AR5)")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# (2) Histogramme des erreurs
axs[1].hist(errors, bins=20, color='gray', edgecolor='black')
axs[1].set_title("Histogramme des erreurs du modèle AR(5)")
axs[1].set_xlabel("Erreur")
axs[1].set_ylabel("Fréquence")
axs[1].grid(True, alpha=0.3)

# (3) ACF des erreurs (10 premiers délais)
plot_acf(errors, lags=10, alpha=0.05, ax=axs[2])
axs[2].set_title("ACF des erreurs (10 premiers délais)")
axs[2].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
    

# ---------------------------------------------------------
# Figure détaillée pour le modèle AR(5) avec NSE
# ---------------------------------------------------------

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf

k = 5
y = soi

# -------- 1. Ajustement du modèle AR(5)
model = AutoReg(y, lags=k, old_names=False).fit()

# Prédictions à partir du lag k
y_pred = model.predict(start=k, end=len(y)-1)
y_true = y[k:]

# -------- 2. Calcul du NSE
numerator = np.sum((y_true - y_pred)**2)
denominator = np.sum((y_true - np.mean(y_true))**2)
nse = 1 - numerator / denominator

print(f"NSE du modèle AR(5) : {nse:.4f}")

# Erreurs
errors = y_true - y_pred

# -------- 3. Figure à 3 sous-axes
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
fig.suptitle(f"Modèle AR(5) – Analyse basée sur le NSE = {nse:.4f}", fontsize=14)

# (1) Série originale vs simulée AR(5)
axs[0].plot(y, label='Série originale')
axs[0].plot(np.arange(k, len(y)), y_pred, label='Série simulée AR(5)')
axs[0].set_title("Série Originale vs Simulée (AR5)")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# (2) Histogramme des erreurs
axs[1].hist(errors, bins=20, color='gray', edgecolor='black')
axs[1].set_title("Histogramme des erreurs du modèle AR(5)")
axs[1].set_xlabel("Erreur")
axs[1].set_ylabel("Fréquence")
axs[1].grid(True, alpha=0.3)

# (3) ACF des erreurs (10 premiers délais)
plot_acf(errors, lags=10, alpha=0.05, ax=axs[2])
axs[2].set_title("ACF des erreurs (10 premiers délais)")
axs[2].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ---------------------------------------------------------
# Comparaison des métriques RMSE, MAE, Biais, NSE pour AR(5)
# ---------------------------------------------------------

from statsmodels.tsa.ar_model import AutoReg

k = 5
y = soi

# Modèle AR(5)
model = AutoReg(y, lags=k, old_names=False).fit()

# Prédictions
y_pred = model.predict(start=k, end=len(y)-1)
y_true = y[k:]

# -------- Calcul des métriques --------
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
mae  = np.mean(np.abs(y_true - y_pred))
bias = np.mean(y_true - y_pred)
nse  = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

metrics = [rmse, mae, bias, nse]
labels  = ["RMSE", "MAE", "Biais", "NSE"]

# ---------------------------------------------------------
# Figure comparative
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(labels, metrics, color=["blue", "orange", "green", "red"])
plt.title("Comparaison des métriques pour le modèle AR(5)")
plt.ylabel("Valeur")
plt.grid(axis='y', alpha=0.3)
plt.show()

# Affichage console
print("\n--- Comparaison des métriques AR(5) ---")
print(f"RMSE  : {rmse:.4f}")
print(f"MAE   : {mae:.4f}")
print(f"Biais : {bias:.4f}")
print(f"NSE   : {nse:.4f}")


# ---------------------------------------------------------
# Scores RMSE, MAE, Biais, NSE et paramètres ak pour AR(5)
# ---------------------------------------------------------

from statsmodels.tsa.ar_model import AutoReg

k = 5
y = soi

# --- Ajustement du modèle AR(5)
model = AutoReg(y, lags=k, old_names=False).fit()

# --- Prédictions
y_pred = model.predict(start=k, end=len(y)-1)
y_true = y[k:]

# --- Calcul des scores
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
mae  = np.mean(np.abs(y_true - y_pred))
bias = np.mean(y_true - y_pred)
nse  = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

# --- Paramètres AR(5)
ak = model.params[1:]    # coefficients sans l'intercept

# --- Affichage console
print("\n===== MODELE AR(5) =====")
print(f"Paramètres ak : {ak.tolist}")
print(f"RMSE  : {rmse:.4f}")
print(f"MAE   : {mae:.4f}")
print(f"Biais : {bias:.4f}")
print(f"NSE   : {nse:.4f}")

# ---------------------------------------------------------
# COMMENTAIRES / JUSTIFICATION DU CHOIX DU MODELE
# ---------------------------------------------------------

# Le modèle AR(5) est analysé ici à l'aide de quatre métriques :
# - RMSE mesure l'erreur quadratique moyenne : plus il est faible,
#   plus les prédictions sont proches de la série observée.
#
# - MAE mesure l'erreur absolue moyenne : c'est une mesure robuste
#   qui indique l'écart moyen entre la série simulée et la série réelle.
#
# - Le biais indique si le modèle a tendance à surestimer (biais > 0)
#   ou sous-estimer (biais < 0) la série. Un biais proche de 0 est idéal.
#
# - Le NSE mesure la capacité du modèle à reproduire la dynamique 
#   de la série. Un NSE proche de 1 indique un très bon ajustement.
#
# Si l'on compare ces quatre critères en même temps, AR(5) est souvent
# un bon compromis : suffisamment flexible pour capturer la dépendance
# temporelle, sans surajuster le bruit. Il fournit généralement 
# un RMSE et un MAE raisonnablement bas, un biais proche de zéro, 
# et un NSE acceptable.
#
# Selon ces critères, AR(5) peut donc être retenu comme modèle 
# représentatif pour cette série.
