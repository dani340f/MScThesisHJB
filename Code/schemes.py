import numpy as np

""" This contains the discretization schemes used for the Monte Carlo methods """


# Euler-Maruyama scheme
def EM(W, X, pi, r, kappa, theta, xi, sigma, A, rho, rho_root, dt, dZ1, dZ2):
    W = W * np.exp((r + 0.25 * pi * (xi - 0.5 * pi) * X**2) * dt + pi * 0.5 * X * (rho * dZ1 + rho_root * dZ2))
    X = ((X + sigma * dZ1) + np.sqrt((X + sigma * dZ1) ** 2 + 8 * kappa * dt * (0.5 * kappa * dt + 1) * A))/(kappa * dt + 2)
    
    return W, X

# Milstein scheme
def Mil(W, X, pi, r, kappa, theta, xi, sigma, A, rho, rho_root, dt, dZ1, dZ2):
    X_next = ((X + sigma * dZ1) + np.sqrt((X + sigma * dZ1) ** 2 + 8 * kappa * dt * (0.5 * kappa * dt + 1) * A))/(kappa * dt + 2)
    W = W * np.exp((r + 0.25 * pi * (xi - 0.5 * pi) * X_next**2) * dt + pi * 0.5 * X * (rho * dZ1 + rho_root * dZ2)
                   + pi * 0.25 * sigma * (rho * (dZ1**2 - dt) + rho_root * dZ1 * dZ2))
    
    return W, X_next

# Two step schemes used for the MLMC method 
def Mil_TwoStep(W, X, pi, r, kappa, theta, xi, sigma, A, rho, rho_root, dt, dZ1, dZ2):
    W, X = Mil(W, X, pi[0], r, kappa, theta, xi, sigma, A, rho, rho_root, dt, dZ1[:,0], dZ2[:,0])
    W, X = Mil(W, X, pi[1], r, kappa, theta, xi, sigma, A, rho, rho_root, dt, dZ1[:,1], dZ2[:,1])
    
    return W, X

def Mil_TwoStep_Antithetic(W, X, pi, r, kappa, theta, xi, sigma, A, rho, rho_root, dt, dZ1, dZ2):
    W[:,0], X[:,0] = Mil_TwoStep(W[:,0], X[:,0], pi, r, kappa, theta, xi, sigma, A, rho, rho_root, dt, dZ1, dZ2)
    W[:,1], X[:,1] = Mil_TwoStep(W[:,1], X[:,1], pi, r, kappa, theta, xi, sigma, A, rho, rho_root, dt, dZ1[:,[1,0]], dZ2[:,[1,0]])
    
    return W, X
