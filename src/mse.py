import numpy as np 

def mse(Y, Y_pred):
    return np.mean((Y - Y_pred)**2)

def mse_prime(Y, Y_pred):
    return -2 * (Y - Y_pred) / np.size(Y)