import numpy as np
from schemes import EM
import time

""" Implementation of standard Monte Carlo """

def stdMC(w0, v0, pi, r, kappa, theta, xi, sigma, rho, T, N, epsilon):
    
    # Step size
    h = T/len(pi)
    
    sqrt_h = np.sqrt(h)
    
    # Constants
    rho_root = np.sqrt(1 - rho**2)
    A = (theta - sigma ** 2/(4*kappa))

    
    cost = 0
    while True:

        # Initialize
        # Variance of mean and bias
        var_mean = np.inf 
        bias = np.inf
        
        # Keep track of number of paths 
        N_save = 0
        
        # Keeps track on the sum and sum of squares to be used when calculating mean and variance
        sums = np.zeros(shape=(3,1))
        
        while var_mean > 0.5 * epsilon**2:
            
            # Initialize
            W = W_DS = w0
            X = X_DS = 2 * np.sqrt(v0)
            hd = h * 2
            
            for j in range(0,int(len(pi)/2)):
                # Sample Wiener increments
                dZ1 = np.random.normal(0, sqrt_h, size=(N,2))
                dZ2 = np.random.normal(0, sqrt_h, size=(N,2))
            
                for i in range(0,2):
                    W, X = EM(W, X, pi[2*j+i], r, kappa, theta, xi, sigma, A, rho, rho_root, h, dZ1[:,i], dZ2[:,i])
                        
                # Sum columns of dZ1 and dZ2
                dZ1 = dZ1.sum(axis=1)
                dZ2 = dZ2.sum(axis=1)
                
                # Calculate double step-size processes
                W_DS, X_DS = EM(W, X, pi[2*j], r, kappa, theta, xi, sigma, A, rho, rho_root, hd, dZ1[:], dZ2[:])
                    
            # Update number of paths simulated       
            N_save += N
            
            # Update sums
            sums[0] += np.sum(W - W_DS)
            sums[1] += np.sum(W)
            sums[2] += np.sum(W ** 2)
            
            # Calculate mean
            mean = sums[1]/N_save
            
            # Calculate variance and variance of mean
            var = (sums[2,:]/(N_save-1) - mean**2*(N_save/(N_save-1)))
            var_mean = var/N_save
            
        # Update computation cost    
        cost += 2 * N_save * len(pi)*1.5
        
        # Calculate bias
        bias = ((sums[0]/N_save)/(1-2))**2
        
        # Check if converted otherwise restart with half step size
        if bias > 0.5*epsilon**2:
            # Half the step size 
            pi = np.repeat(pi, 2)
            h = T/len(pi)
            sqrt_h = np.sqrt(h)
            
        else: 
            break
    
    # Return expectation and standard deviation of terminal wealth
    EW_T = np.round(mean,4)
    stdW_T = np.round(np.sqrt(var),4)
    
    return(EW_T, stdW_T, cost)

if __name__ == "__main__":
    # Final time point
    T = 10
    # Initial wealth
    w0 = 100
    # Initial value of V(T) 
    v0 = 0.0457
    # Riskless interest rate
    r = 0.03
    # Epsilon
    epsilon = 5*10**-1
    # Initial number of paths
    N = 1000
    # Correlation of Wiener processes
    rho = -0.767
    # Kappa
    kappa = 5.07
    # Theta
    theta = 0.0457
    # Xi 
    xi = 1.605
    # Sigma
    sigma = 0.48
    # Optimal control  
    pi = np.array([0.1,0.2,0.15,0.20])
    
    t_start = time.time()
    print(stdMC(w0, v0, pi, r, kappa, theta, xi, sigma, rho, T, N, epsilon)[:2])
    t_end= time.time()
    print("\nRun time: ", round(t_end-t_start,5), "sec")
