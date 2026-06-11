# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:08:29 2026

@author: Administrateur
"""


import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import requests
import json



# ---------------------------------------------------------
# Download AO data from URL and save it as ao.json
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


# Sauvegarde dans ao.json
with open('ao.json', 'w') as file:
    json.dump(json_data, file, indent=4)

print(json.dumps(json_data, indent=4))



# ---------------------------------------------------------
# Affichage lisible des données AO par année
# ---------------------------------------------------------

print("\n===== Données AO triées par année =====\n")

# Tri des dates AAAAMM dans l'ordre croissant
sorted_keys = sorted(data_dict.keys())

current_year = None

for key in sorted_keys:
    year = key[:4]
    month = key[4:]
    value = data_dict[key]

    # Lorsqu'on change d'année, on affiche un nouveau bloc
    if year != current_year:
        if current_year is not None:
            print()  # ajout d'une ligne vide entre les années
        print(f"Année {year} :")
        current_year = year

    # Affichage mensuel
    print(f"   Mois {month} : {value}")


# ---------------------------------------------------------
# Read data from JSON file (exactly comme ton script)
# ---------------------------------------------------------

with open('ao.json', 'r') as file:
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
# Tableau des scores et paramètres AR(k) pour k = 1..5 (DataFrame corrigé)
# ---------------------------------------------------------

# ---------------------------------------------------------
# Tableau des scores et paramètres AR(k) pour k = 1..5 (DataFrame corrigé)
# ---------------------------------------------------------

import pandas as pd

kmax = 5
results = []

for k in range(1, kmax + 1):

    model = AutoReg(y, lags=k, old_names=False).fit()
    y_pred = model.predict(start=k, end=len(y)-1)
    y_true = y[k:]
    
    # Scores
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae  = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_true - y_pred)
    nse  = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

    # Paramètres du modèle
    params = model.params
    param_names = model.model.exog_names   # <-- CORRECTION

    # Ligne du tableau
    row = {
        "k": k,
        "RMSE": rmse,
        "MAE": mae,
        "Biais": bias,
        "NSE": nse
    }

    # Ajout des paramètres au tableau
    for name, value in zip(param_names, params):
        row[name] = value

    results.append(row)

# Construction du dataframe final
df_ar = pd.DataFrame(results)

print("\n==================== DATAFRAME AR(1..5) ====================")
print(df_ar)
print("===========================================================\n")


# ---------------------------------------------------------
# Filtre passe-bas (fréquences > 0.3 supprimées)
# ---------------------------------------------------------
from scipy.signal import butter, filtfilt
cutoff = 0.3        # fréquence de coupure normalisée
order = 4           # ordre du filtre Butterworth

# Création du filtre
b, a = butter(order, cutoff, btype='low', analog=False)

# Application du filtre
soi_filtered = filtfilt(b, a, soi)

# ---------------------------------------------------------
# Visualisation du signal filtré
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(time, soi, label="Original", alpha=0.5)
plt.plot(time, soi_filtered, label="Filtré (fc=0.3)", linewidth=2)
plt.title("Filtrage passe-bas (Butterworth, cutoff = 0.3)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



# ---------------------------------------------------------
# ACF de la série filtrée (25 premiers lags, IC 95%)
# ---------------------------------------------------------

import statsmodels.api as sm

fig, ax = plt.subplots(figsize=(10, 6))
sm.graphics.tsa.plot_acf(soi_filtered, lags=25, alpha=0.05, ax=ax)
ax.set_title("ACF de la série filtrée (Butterworth, fc=0.3)")
ax.set_xlabel("Délais (lags)")
ax.set_ylabel("Autocorrélation")
ax.grid(True, alpha=0.3)
plt.show()


# ---------------------------------------------------------
# Commentaires : Différence entre l’ACF originale et l’ACF filtrée
# ---------------------------------------------------------

# L’ACF de la série originale montrait :
# - une autocorrélation faible au lag 1 (~0.25),
# - puis une chute rapide dans l’intervalle de confiance.
#
# Cela indique que la série AO brute est dominée par des
# fluctuations rapides, de type bruit quasi blanc, avec très
# peu de mémoire temporelle : seule la valeur du mois
# précédent a une influence modérée.
#
# Après application du filtre passe-bas Butterworth (fc = 0.3),
# l’ACF change complètement de structure :
#
# - l’autocorrélation au lag 1 devient très élevée (~0.95),
# - les lags 2 à 5 montrent aussi des valeurs significatives,
# - puis la décroissance est lente et régulière.
#
# Cette différence s’explique par le filtrage :
# en supprimant les hautes fréquences, le filtre élimine
# les variations rapides et rend la série beaucoup plus lisse.
# Une série lissée est nécessairement beaucoup plus corrélée
# d’un mois sur l’autre, ce qui produit des autocorrelations
# élevées sur plusieurs lags.
#
# En résumé :
# - ACF originale : faible persistance → comportement bruité.
# - ACF filtrée : forte persistance → comportements lents,
#   typiques des basses fréquences conservées par le filtre.
#
# Donc, l'ACF filtrée révèle la structure lente du signal AO
# qui était masquée par le bruit haute fréquence dans la
# série originale.