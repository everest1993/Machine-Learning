"""
Logica per il calcolo della regressione lineare univariata:
- compute_gradient per il calcolo del gradiente dati i valori w e b
- gradient_descent per eseguire il metodo di discesa del gradiente e individuare i minimi
"""

def compute_gradient(x, y, w, b):
    """
    Calcola il gradiente per la regressione lineare
    Parametri:
        x : input (m elementi)
        y : valori effettivi
        w, b: coefficiente angolare e intercetta
    Restituisce:
        dj_dw: gradiente per il parametro w
        dj_db: gradiente per il parametro b
    """
    m = x.shape[0]
    dj_dw = 0 # derivata parziale rispetto a w
    dj_db = 0 # derivata parziale rispetto a b

    for i in range(m):
        f_wb = w * x[i] + b # applicazione del modello lineare

        # gradienti come derivate dell'errore quadratico medio
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i]

        dj_dw += dj_dw_i
        dj_db += dj_db_i

    # i gradienti rappresentano la pendenza della superficie di costo J(w, b)
    # per individuare il minimo è necessario muoversi sulla superficie in direzione contraria
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, gradient_function): 
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