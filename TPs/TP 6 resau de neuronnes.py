# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 05:25:49 2026

@author: Administrateur
"""

# ==========================================================
# TP MODULE 6 - MODELISATION NON LINEAIRE
# Réseau de neurones avec Keras / TensorFlow

# ==========================================================
# OBJECTIF :
# Prédire le rayonnement incident de courtes longueurs d’onde
# à partir des variables météo de mesonet.txt
# ==========================================================


# ==========================================================
# ETAPE 1 - IMPORTATION DES LIBRAIRIES
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Nadam


# ==========================================================
# ETAPE 2 - CHARGEMENT DES DONNEES
# ==========================================================

fichier = r"C:\Users\Administrateur\OneDrive - Université Laval\mesonet.txt"

data = np.loadtxt(fichier)

# cible
y = data[:, 0]

# variables explicatives
X = data[:, 1:6]


# Colonnes :
# 0 = cible = rayonnement incident
# 1 = jour de l'année
# 2 = rayonnement hors atmosphère
# 3 = Tmax
# 4 = Tmin
# 5 = écart Tmax-Tmin

# Variable cible
y = data[:, 0]

# Variables explicatives
X = data[:, 1:6]


# ==========================================================
# ETAPE 3 - DIVISION DES DONNEES
# 70% entrainement
# 15% validation
# 15% test
# ==========================================================

# Première séparation : train 70% / reste 30%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Deuxième séparation : validation 15% / test 15%
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)


# ==========================================================
# ETAPE 4 - NORMALISATION DES DONNEES
# Très important pour les réseaux de neurones
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)


# ==========================================================
# ETAPE 5 - FONCTION POUR CREER UN MODELE
# ==========================================================

def creer_modele(nb_neurones1=16,
                 nb_neurones2=8,
                 activation='relu'):
    
    model = Sequential()
    
    # Couche cachée 1
    model.add(Dense(nb_neurones1,
                    activation=activation,
                    input_shape=(5,)))
    
    # Couche cachée 2
    model.add(Dense(nb_neurones2,
                    activation=activation))
    
    # Sortie (1 valeur continue)
    model.add(Dense(1, activation='linear'))
    
    # Compilation
    model.compile(
        optimizer=Nadam(),
        loss='mse'
    )
    
    return model


# ==========================================================
# ETAPE 6 - TEST DE PLUSIEURS MODELES
# ==========================================================

activations = ['relu', 'elu', 'tanh', 'softsign', 'linear']

resultats = []

for act in activations:
    
    print("Test activation :", act)
    
    model = creer_modele(
        nb_neurones1=16,
        nb_neurones2=8,
        activation=act
    )
    
    historique = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        verbose=0
    )
    
    # Prédiction validation
    pred_val = model.predict(X_val, verbose=0)
    
    mse_val = mean_squared_error(y_val, pred_val)
    
    resultats.append([act, mse_val])
    
    print("MSE validation =", mse_val)


# ==========================================================
# ETAPE 7 - TABLEAU DES RESULTATS
# ==========================================================

df_resultats = pd.DataFrame(
    resultats,
    columns=['Activation', 'MSE_validation']
)

print(df_resultats)

# Meilleure activation
meilleur_act = df_resultats.sort_values(
    by='MSE_validation'
).iloc[0, 0]

print("\nMeilleure activation :", meilleur_act)


# ==========================================================
# ETAPE 8 - MODELE FINAL
# Train + Validation combinés
# ==========================================================

X_final = np.vstack((X_train, X_val))
y_final = np.concatenate((y_train, y_val))

modele_final = creer_modele(
    nb_neurones1=16,
    nb_neurones2=8,
    activation=meilleur_act
)

historique_final = modele_final.fit(
    X_final,
    y_final,
    epochs=200,
    verbose=0
)


# ==========================================================
# ETAPE 9 - EVALUATION FINALE SUR TEST
# ==========================================================

pred_test = modele_final.predict(X_test, verbose=0)

mse_test = mean_squared_error(y_test, pred_test)
r2 = r2_score(y_test, pred_test)

print("\n==============================")
print("RESULTATS FINAUX")
print("==============================")
print("MSE test =", mse_test)
print("R2 =", r2)


# ==========================================================
# ETAPE 10 - GRAPHIQUE APPRENTISSAGE
# ==========================================================

plt.figure(figsize=(8,5))
plt.plot(historique.history['loss'], label='Train')
plt.plot(historique.history['val_loss'], label='Validation')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Courbe d'apprentissage")
plt.legend()
plt.grid()
plt.show()


# ==========================================================
# ETAPE 11 - GRAPHIQUE REEL VS PREVU
# ==========================================================

plt.figure(figsize=(7,7))
plt.scatter(y_test, pred_test, alpha=0.7)

plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    'r--'
)

plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Réel vs Prédit")
plt.grid()
plt.show()