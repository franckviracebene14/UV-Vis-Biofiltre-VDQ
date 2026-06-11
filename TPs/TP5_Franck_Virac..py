


# IMPORT DES LIBRAIRIES


import matplotlib.pyplot as plt     
import numpy as np 
import pandas as pd                 
import statsmodels.api as sm        
from statsmodels.tsa.ar_model import AutoReg 
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error 
from scipy.signal import butter, filtfilt
import requests                      
import json                         


# # # TÉLÉCHARGEMENT DES DONNÉES AO (Arctic Oscillation)


url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table"

# Récupération du contenu brut du fichier
text = requests.get(url).text

# Découpage en lignes
lines = text.strip().split("\n")

# Dictionnaire pour stocker les données
data_dict = {}

# Parcours des lignes (on saute l'en-tête)
for line in lines[1:]:
    parts = line.split()     # séparation colonnes
    year = parts[0]          # année
    values = parts[1:]       # valeurs mensuelles
    
    # Boucle sur les 12 mois
    for i, v in enumerate(values):
        month = str(i+1).zfill(2)   # format "01", "02", ...
        key = year + month          # format AAAAMM
        data_dict[key] = float(v)   # stockage


# Structure JSON finale
json_data = {"data": data_dict}

# Sauvegarde dans un fichier
with open('ao.json', 'w') as file:
    json.dump(json_data, file, indent=4)

# Affichage du JSON
print(json.dumps(json_data, indent=4))



# # # AFFICHAGE DES DONNÉES PAR ANNÉE


print("Données AO triées par année")

# Tri des clés temporelles
sorted_keys = sorted(data_dict.keys())

current_year = None

# Parcours des données
for key in sorted_keys:
    year = key[:4]      # extraction année
    month = key[4:]     # extraction mois
    value = data_dict[key]

    # Détection changement d’année
    if year != current_year:
        if current_year is not None:
            print()
        print(f"Année {year} :")
        current_year = year

    print(f"   Mois {month} : {value}")



# # # LECTURE DES DONNÉES JSON


with open('ao.json', 'r') as file:
    soi_dict = json.load(file)

data = soi_dict['data']

# Création des tableaux numpy
soi = np.zeros(len(data))   # valeurs AO
time = np.zeros(len(data))  # temps en années décimales

# Conversion des données
count = 0
for key, value in data.items():
    soi[count] = float(value)
    # Conversion AAAAMM -> année décimale
    time[count] = float(key[:4]) + (float(key[4:]) - 1) / 12
    count += 1



# # # VISUALISATION DE LA SÉRIE TEMPORELLE

plt.figure(figsize=(10, 6))
plt.plot(time, soi, label='Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Artic oscillation')
plt.legend()
plt.show()



# # # AUTOCORRÉLATION (ACF)


fig, ax = plt.subplots(figsize=(10, 6))
sm.graphics.tsa.plot_acf(soi, lags=25, alpha=0.05, ax=ax)
plt.title('Autocorrelogram with 95% Confidence Interval')
plt.show()



# # # ÉVALUATION DES MODÈLES AR(k)

kmax = 25

rmse_vals = []
mae_vals = []
bias_vals = []
nse_vals = []

y = soi  # série étudiée

# Boucle sur les ordres AR
for k in range(1, kmax + 1):
    model = AutoReg(y, lags=k, old_names=False).fit()
    
    y_pred = model.predict(start=k, end=len(y)-1)
    y_true = y[k:]
    
    # Calcul des métriques
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae  = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_true - y_pred)
    nse  = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    
    rmse_vals.append(rmse)
    mae_vals.append(mae)
    bias_vals.append(bias)
    nse_vals.append(nse)



# # # VISUALISATION DES PERFORMANCES


fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
k_vals = np.arange(1, kmax+1)

axs[0].plot(k_vals, rmse_vals, marker='o')
axs[0].set_ylabel("RMSE")
axs[0].grid(True)

axs[1].plot(k_vals, mae_vals, marker='o', color='orange')
axs[1].set_ylabel("MAE")
axs[1].grid(True)

axs[2].plot(k_vals, bias_vals, marker='o', color='green')
axs[2].set_ylabel("Bias")
axs[2].grid(True)

axs[3].plot(k_vals, nse_vals, marker='o', color='red')
axs[3].set_ylabel("NSE")
axs[3].set_xlabel("k (lag)")
axs[3].grid(True)

plt.suptitle("Performance des modèles AR(k)")
plt.show()



# # # ANALYSE DÉTAILLÉE DES MODÈLES AR(1..5)


kmax = 5

for k in range(1, kmax + 1):

    model = AutoReg(y, lags=k, old_names=False).fit()

    y_pred = model.predict(start=k, end=len(y)-1)
    y_true = y[k:]
    errors = y_true - y_pred

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # Série réelle vs prédite
    axs[0].plot(y, label='Original')
    axs[0].plot(np.arange(k, len(y)), y_pred, label=f'AR({k})')
    axs[0].legend()

    # Histogramme des erreurs
    axs[1].hist(errors, bins=20)

    # ACF des erreurs
    plot_acf(errors, lags=10, ax=axs[2])

    plt.tight_layout()
    plt.show()



# # # TABLEAU DES RÉSULTATS


results = []

for k in range(1, 6):

    model = AutoReg(y, lags=k, old_names=False).fit()
    y_pred = model.predict(start=k, end=len(y)-1)
    y_true = y[k:]
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae  = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_true - y_pred)
    nse  = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

    row = {"k": k, "RMSE": rmse, "MAE": mae, "Biais": bias, "NSE": nse}

    # Ajout des coefficients du modèle
    for name, value in zip(model.model.exog_names, model.params):
        row[name] = value

    results.append(row)

df_ar = pd.DataFrame(results)

print(df_ar)


# # # Commentaire: quel modèle choisir?


# Il ya une seule variable qui sort de l'intervale de confiance
# l'ACF de la serie montre uniquement une autocorrelation significative avec la premiere variable (lag) ordre 1
# À partir du modèle 2 jusqu'à 5 les scores (RMSE, MAE, NSE) ne varient presque pas.
# Ainsi nous choisissons le modele 1 (AR1)




# # # FILTRE PASSE-BAS (Butterworth)



cutoff = 0.3   # fréquence de coupure
order = 4      # ordre du filtre

# Création du filtre
b, a = butter(order, cutoff, btype='low')

# Application du filtre
soi_filtered = filtfilt(b, a, soi)



# # # VISUALISATION DU SIGNAL FILTRÉ


plt.figure(figsize=(10, 6))
plt.plot(time, soi, label="Original", alpha=0.5)
plt.plot(time, soi_filtered, label="Filtré")
plt.legend()
plt.show()



# # # ACF APRÈS FILTRAGE


fig, ax = plt.subplots(figsize=(10, 6))
sm.graphics.tsa.plot_acf(soi_filtered, lags=25, alpha=0.05, ax=ax)
plt.show()



# COMMENTAIRE FINAL:Différence entre l’ACF originale et l’ACF filtrée


# L’ACF de la série originale montrait :
# une autocorrélation faible au lag 1 (0.25),
# puis une chute rapide dans l’intervalle de confiance.

# Cela indique que la série AO brute est dominée par des
# fluctuations rapides, de type bruit, avec trèspeu de mémoire temporelle : seule la valeur du mois
# précédent a une influence modérée.
# Après l'application du filtre passe-bas Butterworth (fc = 0.3),
# l’ACF change complètement de structure :

# on note trois variables (lag) qui sortent de l'intervalle de confiance
# - l’autocorrélation au lag 1 devient très élevée (0.95),
# - les lags 2 et 3 montrent aussi des valeurs significatives,

# Cette différence s’explique par le filtrage :
# en supprimant les hautes fréquences, le filtre élimine
# les variations rapides et rend la série beaucoup plus lisse.
# on remarque qu'une série lissée est beaucoup plus corrélée
# d’un mois sur l’autre, ce qui produit des autocorrelations
# élevées sur plusieurs lags.


