"""
Logica per il calcolo della regressione lineare univariata:
- compute_gradient per il calcolo del gradiente dati i valori w e b
- gradient_descent per eseguire il metodo di discesa del gradiente e individuare i minimi
"""

from typing import Tuple
import numpy as np


def compute_gradient(
        x: np.ndarray, 
        y: np.ndarray, 
        w: float, 
        b: float
        ) -> Tuple[float, float]:
    """
    Calcola il gradiente per la regressione lineare
    Parametri:
        x: input (m elementi)
        y: valori effettivi
        w, b: coefficiente angolare e intercetta
    Restituisce:
        dj_dw: gradiente per il parametro w
        dj_db: gradiente per il parametro b
    """
    m = x.shape[0]
    dj_dw = 0.0 # derivata parziale rispetto a w
    dj_db = 0.0 # derivata parziale rispetto a b

    for i in range(m):
        f_wb = w * x[i] + b # applicazione del modello lineare

        # gradienti come derivate dell'errore quadratico medio
        dj_dw += np.dot((f_wb - y[i]), x[i])
        dj_db += f_wb - y[i]

    return dj_dw / m, dj_db / m


def gradient_descent(
        x: np.ndarray, 
        y: np.ndarray, 
        w_in: float, 
        b_in: float, 
        alpha: float, 
        num_iters: int, 
        gradient_function
        ) -> Tuple[float, float]: 
    """
    Esegue il metodo di discesa del gradiente per i parametri w e b. 
    Aggiorna w, b effettuando num_iters steps con learning rate alpha
    Parametri:
        x: input (m elementi)
        y: valori effettivi
        w_in, b_in: valori iniziali del modello
        alpha: learning rate
        num_iters: numero di iterazioni per calcolare la discesa del gradiente
        gradient_function: funzione per il calcolo del gradiente
    Restituisce:
        w: punto di minimo per il coefficiente angolare
        b: punto di minimo per l'intercetta
    """
    w = w_in
    b = b_in

    for i in range(num_iters):
        # calcolo del gradiente nei punti w e b della superficie
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # aggiornamento dei parametri sottraendo il gradiente (direzione contraria alla pendenza)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
 
    return w, b