import numpy as np
from stdMC import stdMC
from MLMC import MLMC
import matplotlib.pyplot as plt
plt.ion()

""" Reproduces the plots of section 5.1 """

np.random.seed(0)


marker_styles = ['d','*','s','o']
colors = ['tab:blue', 'tab:orange', 'tab:green','tab:red']
plt.style.use('bmh')

""" Parameters """
# Final time point
T = 10
# Initial wealth
w0 = 100
# Initial value of V(T) 
v0 = 0.0457
# Riskless interest rate
r = 0.03
# Epsilon
epsilon = 5*10**-2
# Initial number of steps
M = 10
# Initial number of paths
N = 10
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
pi = np.array([0.1])

pis = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])


epsilons = np.linspace(10**-1, 10**-2, num=5)
cost_MLMC_av = np.zeros(len(epsilons))
cost_MLMC = np.zeros(len(epsilons))
cost_stdMC = np.zeros(len(epsilons))

for i, epsilon in enumerate(epsilons):
    MLMC_av_save = MLMC(w0, v0, pi, r, kappa, theta, xi, sigma, rho, T, N, epsilon, True)
    MLMC_save = MLMC(w0, v0, pi, r, kappa, theta, xi, sigma, rho, T, N, epsilon, False)
    stdMC_save = stdMC(w0, v0, pis, r, kappa, theta, xi, sigma, rho, T, 1000, epsilon)
    
    # save cost
    cost_MLMC_av[i] = MLMC_av_save[2]
    cost_MLMC[i] = MLMC_save[2]
    cost_stdMC[i] = stdMC_save[2]
    print('hej')

""" Figure 1 """
N_l_av, varMLMCav = MLMC_av_save[3:5]
N_l, varMLMC, var_l_MLMC = MLMC_save[3:6]
plt.figure(dpi=300)
plt.xlabel(r'level $l$', fontsize=15)
plt.ylabel(r'$\log_2$ variance ', fontsize=15)
plt.plot(np.arange(len(var_l_MLMC)), np.log2(var_l_MLMC),'-o', color = colors[2], label = r'$f_l$')
plt.plot(np.arange(1,len(varMLMC)), np.log2(varMLMC[1:]),'-s', color = colors[0], label=r'$f_l-f_{l-1}$')
plt.plot(np.arange(1,len(varMLMCav)), np.log2(varMLMCav[1:]),'-o', color = colors[3], label=r'$f_l^{av}-f_{l-1}$')
plt.legend()

# Slopes
# (np.log2(varMLMC[-1])-np.log2(varMLMC[-5]))/4
# (np.log2(varMLMCav[-1])-np.log2(varMLMCav[-5]))/4

""" Figure 2 """
plt.figure(dpi=300)
plt.xlabel(r'level $l$', fontsize=15)
plt.ylabel(r'$N_l$', fontsize=15)
plt.plot(np.arange(len(N_l)), np.log2(N_l),'-s', color = colors[0], label=r'$N_l$')
plt.plot(np.arange(len(N_l_av)), np.log2(N_l_av),'-o', color = colors[3], label=r'$N_l^{av}$')
plt.legend()

# Slopes
# (np.log2(N_l[-1])-np.log2(N_l[-7]))/6
# (np.log2(N_l_av[-1])-np.log2(N_l_av[-7]))/6
""" Figure 3 """
plt.figure(dpi=300)
plt.xlabel(r'$\epsilon$', fontsize=15)
plt.ylabel(r'$\epsilon^2$cost', fontsize=15)
plt.loglog(epsilons, epsilons**2 * cost_MLMC_av, '-o',color = colors[3], label ='MLMC (av)')
plt.loglog(epsilons, epsilons**2 * cost_MLMC, '-s',color = colors[0], label ='MLMC')
plt.loglog(epsilons, epsilons**2 * cost_stdMC, '-*',color = colors[1], label ='std. MC')
plt.legend()

plt.tight_layout()

