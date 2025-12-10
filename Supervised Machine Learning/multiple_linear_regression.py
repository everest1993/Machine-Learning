"""
Logica per il calcolo della regressione lineare multipla
- compute_gradient per il calcolo del gradiente dati i valori w e b
- gradient_descent per eseguire il metodo di discesa del gradiente e individuare i minimi
"""

from typing import Tuple
import numpy as np


def compute_gradient(
        X: np.ndarray, 
        y: np.ndarray, 
        w: np.ndarray, 
        b: float
        ) -> Tuple[np.ndarray, float]:
    """
    Calcola il gradiente per la regressione lineare
    Parametri:
        X: input (matrice di dimensione m x n)
        y: array di valori effettivi
        w: array contenente i coefficienti angolari
        b: intercetta
    Restituisce:
        dj_dw: array con gradienti per i parametri w
        dj_db: gradiente per il parametro b
    """
    m, n = X.shape

    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m): # per ogni riga
        err = (np.dot(X[i], w) + b) - y[i] # errore calcolato sul vettore

        for j in range(n): # per ogni colonna
            dj_dw[j] += err * X[i][j] # calcolo gradiente con derivata
        
        dj_db += err

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(
        X: np.ndarray, 
        y: np.ndarray, 
        w_in: np.ndarray, 
        b_in: float, 
        alpha: float, 
        num_iters: int, 
        gradient_function
        ) -> Tuple[np.ndarray, float]: 
    """
    Esegue il metodo di discesa del gradiente per i parametri w e b. 
    Aggiorna w, b effettuando num_iters steps con learning rate alpha
    Parametri:
        X: input (matrice m x n)
        y: array di valori effettivi
        w_in: array con valori iniziali di w
        b_in: valore iniziale di b
        alpha: learning rate
        num_iters: numero di iterazioni per calcolare la discesa del gradiente
        gradient_function: funzione per il calcolo del gradiente
    Restituisce:
        w: array di punti di minimo per il coefficiente angolare
        b: punto di minimo per l'intercetta
    """
    w = np.copy(w_in)
    b = b_in

    for _ in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w, b)
        
        w -= alpha * dj_dw
        b -= alpha * dj_db

    return w, b