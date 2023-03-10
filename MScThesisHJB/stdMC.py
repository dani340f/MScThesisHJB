import numpy as np
from schemes import EM
import time

""" Implementation of standard Monte Carlo """

def stdMC(w0, v0, p_opt, r, kappa, theta, xi, sigma, rho, T, M, N, epsilon):
    
    # Step size
    h = T/M
    sq_h = np.sqrt(h)
    
    # Constants
    rho_root = np.sqrt(1 - rho**2)
    A = (theta - sigma ** 2/(4*kappa))

    # Initialize procedure
    done = False
    
    cost = 0
    while done == False:

        # Initialize
        # Variance of mean and bias
        var_mean = np.inf 
        bias = np.inf
        
        # Keeps track of number of paths 
        N_save = 0
        
        # Keeps track on the sum and sum of squares to be used when calculating mean and variance
        sums = np.zeros(shape=(3,1))
        
        while var_mean > 0.5 * epsilon**2:
            
            # Initialize
            W = W_DS = w0
            X = X_DS = 2 * np.sqrt(v0)
            hd = h * 2
            
            for j in range(0,int(M/2)):
                # Sample Wiener increments
                dZ1 = np.random.normal(0, sq_h, size=(N,2))
                dZ2 = np.random.normal(0, sq_h, size=(N,2))
            
                for i in range(0,2):
                    W, X = EM(W, X, p_opt, r, kappa, theta, xi, sigma, A, rho, rho_root, h, dZ1[:,i], dZ2[:,i])
                        
                # Sum columns of dZ1 and dZ2
                dZ1 = dZ1.sum(axis=1)
                dZ2 = dZ2.sum(axis=1)
                
                # Calculate double step processes
                W_DS, X_DS = EM(W, X, p_opt, r, kappa, theta, xi, sigma, A, rho, rho_root, hd, dZ1[:], dZ2[:])
                    
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
        cost += 2 * N_save * (M+M/2)
        
        # Calculate bias
        bias = ((sums[0]/N_save)/(1-2**1))**2
        
        # Check if converted otherwise restart with half step size
        if bias > 0.5*epsilon**2:
            # Half the step size 
            M *= 2
            h = T/M
            sq_h = np.sqrt(h)
            
        else: 
            done = True
    
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
    epsilon = 10**-1
    # Initial number of steps
    M = 10
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
    p_opt = 0.1
    
    t_start = time.time()
    print(stdMC(w0, v0, p_opt, r, kappa, theta, xi, sigma, rho, T, M, N, epsilon)[:2])
    t_end= time.time()
    print("\nRun time: ", round(t_end-t_start,5), "sec")
