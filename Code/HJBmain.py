import numpy as np
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import bicgstab, gmres, spsolve
import time


# t_start = time.time()

# t_end= time.time()
# print("\nRun time: ", round(t_end-t_start,5), "sec")


""" This script contains the implementation of 
                              the numerical solution scheme of the HJB equation """

# def HJBmain():
    
""" Parameters """
# Refinment
refinment = 0
# Riskless interest rate
r = 0.03
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
# eta 
eta = 540.0
# Initial conditions
w_0 = 100.0
v_0 = 0.0457

scale = 1.0
tol = 1e-5

""" Wealth discretization """
w_min = 0.0
w_max = 3*10.0**3
# #nodes in wealth direction
N_1 = (111 * 2**refinment) + 1

w = np.linspace(w_min, w_max, N_1)

dw = (w_max-w_min)/(N_1-1)

""" Varinace discretization """
v_min = 0.0
v_max = 2.0
# #nodes in variance direction
N_2 = (56 * 2**refinment) + 1
v = np.linspace(v_min, v_max, N_2)

dv = (v_max-v_min)/(N_2-1)

# #nodes in grid
N = N_1 * N_2

""" Control discretixation """
# Leverage constraint
pi_max = 2.0
# #nodes in control space
J = 8 * 2**refinment
PIs = np.linspace(0.0, pi_max ,J)

""" Time discretixation """
# Terminal date
T = 10.0
# #timesteps
N_tau = 160 * 2**refinment

dtau = T/N_tau

""" Numerical scheme """

# Stencil length
h = dv
# Wide stencil
h_sqrt = np.sqrt(h)

x = np.array([np.tile(w, N_2), v.repeat(N_1)]).T

"""  Relevant domains """
# Inside 
Omega_in = np.where((w_min < x[:,0]) & (x[:,0] < w_max) & (v_min < x[:,1]) & (x[:,1] < v_max))

# It is helpful to split Omega_in into the following two subdomains 
Omega_in_lower = np.where((w_min < x[:,0]) & (x[:,0] < w_max) & (v_min < x[:,1]) & (x[:,1] <= h_sqrt))
Omega_in_upper = np.where((w_min < x[:,0]) & (x[:,0] < w_max) & (h_sqrt < x[:,1]) & (x[:,1] < v_max))

# Upper boundaries
Omega_wmax = np.where(x[:,0] == w_max)
Omega_vmax = np.where((x[:,1] == v_max) & (x[:,0] != w_max))
# Lower boundaries
Omega_wmin = np.where((x[:,0] == w_min) & (x[:,1] != v_max))
Omega_vmin = np.where((x[:,1] == v_min) & (x[:,0] != w_max) & (x[:,0] != w_min))

# We will use the following domains the keep track on indexes inside the subdomains
Omega_in_NST = np.where((h_sqrt < x[Omega_in][:,1]) & (x[Omega_in][:,1] < v_max))
Omega_in_STL = np.where(x[Omega_in][:,1] <= h_sqrt)

def index(i,j):
    return ((i+1) + j * N_1 - 1)

def BC(tau, w): # Asymptotic boundary condition on upper boundary w = w_max
    
    H_0 = eta**2/4
    H_1 = - eta * np.exp(r*tau)
    H_2 = np.exp(2*r*tau)
    
    return H_2 * w**2 + H_1 * w + H_0


""" The wide stencil method """
def phi(controls, x): # Grid roation angle
    
    c1 = (2 * rho * controls * sigma * x[:,0] * x[:,1])/((controls * np.sqrt(x[:,1]) * x[:,0])**2 - (sigma * np.sqrt(x[:,1]))**2)

    return 0.5 * np.arctan(c1)

def a(controls, x, angles): # Wide stencil coefficient "a"
    
    c1 = 0.5 * (controls * np.sqrt(x[:,1]) * x[:,0])**2 * np.cos(angles)**2
    c2 = rho * controls * sigma * x[:,0] * x[:,1] * np.sin(angles) * np.cos(angles)
    c3 = 0.5 * (sigma * np.sqrt(x[:,1]))**2 * np.sin(angles)**2

    return c1 + c2 + c3

def b(controls, x, angles): # Wide stencil coefficient "b"
    
    c1 = 0.5 * (controls * np.sqrt(x[:,1]) * x[:,0])**2 * np.sin(angles)**2
    c2 = rho * controls * sigma * x[:,0] * x[:,1] * np.sin(angles) * np.cos(angles)
    c3 = 0.5 * (sigma * np.sqrt(x[:,1]))**2 * np.cos(angles)**2
    
    return c1 - c2 + c3

def Psi(x, angles): # Points on virtual grid
    # Columns of rotation matrix
    R_1 = np.array([np.cos(angles), np.sin(angles)]).T
    R_2 = np.array([-np.sin(angles), np.cos(angles)]).T
    
    # Four off-grid points
    Psi_1 = x + h_sqrt*(R_1)
    Psi_2 = x - h_sqrt*(R_1)
    Psi_3 = x + h_sqrt*(R_2)
    Psi_4 = x - h_sqrt*(R_2)
    
    return [Psi_1, Psi_2, Psi_3, Psi_4]

# terminal_values = ((x[:,0]-eta/2)**2)
# controls = np.ones_like(terminal_values)
# hej = Psi(x[Omega_in], phi(controls[Omega_in], x[Omega_in]))[3]

""" Degenerate linear operator coefficients """
def alpha_w(controls, x):
    
    return ((controls * np.sqrt(x[:,1]) * x[:,0])**2)/(2*dw**2)


def beta_w(controls, x):
    
    return ((controls * np.sqrt(x[:,1]) * x[:,0])**2)/(2*dw**2)


def alpha_v(x):

    c1 = (sigma * np.sqrt(x[:,1]))**2/(2*dv**2)
    c2 = np.maximum(0, - (kappa*(theta-x[:,1]))/dv)
    
    return c1 + c2


def beta_v(x):
    
    c1 = (sigma * np.sqrt(x[:,1]))**2/(2*dv**2)
    c2 = np.maximum(0, (kappa*(theta-x[:,1]))/dv)

    return c1 + c2


""" Algorithm 4.3: Improved linear interpolation """
def improved_linear_interpolation(x, tau):
    # Find location in grid
    index_left_w = np.searchsorted(w, x[:,0], side="right") - 1 
    index_left_v = np.searchsorted(v, x[:,1], side="right") - 1
    
    # If w = w_max or v = v_max we have to shift the left index one time to the left
    index_left_v[np.where(index_left_v == N_2-1)] -= 1
    index_left_w[np.where(index_left_w == N_1-1)] -= 1
    
    w_left = w[index_left_w]; w_right = w[index_left_w + 1]
    v_left = v[index_left_v]; v_right = v[index_left_v + 1]
    
    # Calculate w_special
    w_special = eta/2 * np.exp(-r*(tau))
    
    # Allocate storage for interpolation coefficients 
    omega_11 = np.zeros_like(x[:,0])
    omega_12 = np.zeros_like(x[:,0])
    omega_21 = np.zeros_like(x[:,0])
    omega_22 = np.zeros_like(x[:,0])
    
    # Case 1: Standard linear interpolation
    case1 = np.where(np.logical_or(w_special < w_left, w_special > w_right))
    
    omega_11[case1] = (w_right[case1] - x[case1][:,0])*(v_right[case1] - x[case1][:,1])/(dw * dv)
    omega_12[case1] = (w_right[case1] - x[case1][:,0])*(x[case1][:,1] - v_left[case1])/(dw * dv)
    omega_21[case1] = (x[case1][:,0] - w_left[case1])*(v_right[case1]-x[case1][:,1]) /(dw * dv)
    omega_22[case1] = (x[case1][:,0] - w_left[case1])*(x[case1][:,1]-v_left[case1]) /(dw * dv)
    
    # Case 2 & 3: Improved linear interpolation  
    case2 = np.where(np.logical_and(w_special <= x[:,0], w_special >= w_left))
    w_left[case2] = w_special
    
    omega_21[case2] = (x[case2][:,0] - w_left[case2])*(v_right[case2]-x[case2][:,1]) /((w_right[case2] - w_left[case2])*dv) 
    omega_22[case2] = (x[case2][:,0] - w_left[case2])*(x[case2][:,1]-v_left[case2]) /((w_right[case2] - w_left[case2])*dv)
    
    case3 = np.where(np.logical_and(w_special > x[:,0], w_special <= w_right))
    w_right[case3] = w_special
    
    omega_11[case3] = (w_right[case3] - x[case3][:,0])*(v_right[case3] - x[case3][:,1]) /((w_right[case3] - w_left[case3])*dv) 
    omega_12[case3] = (w_right[case3] - x[case3][:,0])*(x[case3][:,1] - v_left[case3]) /((w_right[case3] - w_left[case3])*dv)
    
    return omega_11, omega_12, omega_21, omega_22, index_left_w, index_left_v

""" Linear Lagrange interpolation operator """
def PHI(controls, x, U, tau):
    
    PHInp1 =np.zeros_like(controls)
    
    w_star = x[:,0]* np.exp((r + controls * xi * x[:,1]) * dtau)
    
    # Case 1:
    case1 = np.where(w_star >= w_max)
    PHInp1[case1] = BC(tau-dtau,w_star[case1])
    
    # Case 2:
    case2 = np.where(w_star < w_max)
    x = np.array([w_star[case2], x[case2][:,1]]).T
    
    omega_11, omega_12, omega_21, omega_22, index_left_w, index_left_v = improved_linear_interpolation(x, tau-dtau)
    
    PHInp1[case2] = (omega_11 * U[index(index_left_w, index_left_v)] + omega_12 * U[index(index_left_w, index_left_v + 1)]
                    + omega_21 * U[index(index_left_w + 1, index_left_v)] + omega_22 * U[index(index_left_w +1, index_left_v + 1)])
    
    # return x
    return PHInp1
# dav = PHI(controls, x, terminal_values, dtau)

def G(controls, x, tau):
    
    Gnp1 =np.zeros_like(controls)
    
    # Calculate off-grid points
    angles = phi(controls[Omega_in_upper2], x[Omega_in_upper2])
    
    # Caculate of grid points
    Psi_m = Psi(x[Omega_in_upper2[0]], angles)
    
    # Calculate coefficients
    coef_1 = a(controls[Omega_in_upper2], x[Omega_in_upper2], angles)
    coef_2 = b(controls[Omega_in_upper2], x[Omega_in_upper2], angles)
    
    for m in range(4):
        out_of_domain = np.where(Psi_m[m][:,1] > v_max)
        
        if out_of_domain[0].size:
            
            if m < 2:
                coef = coef_1
            else:
                coef = coef_2
            
            Gnp1[Omega_in_upper2[0][out_of_domain]] += coef[out_of_domain]/h * BC(tau, Psi_m[m][out_of_domain][:,0]) 
    
    return Gnp1

# Gnp1 = G(controls, x, dtau)

""" Lnp1 matrix """
" The Lnp1 matrix is a sparse N x N matrix, in order to save memory it is convinient not to store all of the zero-elements. "
" We prevent this by constructing the matrix in a COOrdinarte format, "
"see e.g. https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html "

def L(controls, x, tau):
    " Upper boundary, v = v_max (Note the first index is droped) "
    coef_1 = alpha_w(controls[Omega_vmax][1:], x[Omega_vmax][1:,]); coef_2 = beta_w(controls[Omega_vmax][1:], x[Omega_vmax][1:,])
    
    row = np.tile(Omega_vmax[0][1:], 3)
    column = np.tile(Omega_vmax[0][1:], 3)
    
    # Adjust indexes of columns
    column[:len(Omega_vmax[0][1:])] -=1
    column[2*len(Omega_vmax[0][1:]):] +=1
    
    data = np.concatenate((coef_1, -(coef_1 + coef_2), coef_2), axis=None)
    
    " Lower boundary, w = 0 (Here the first index of coef_1 is droped)"
    coef_1 = alpha_v(x[Omega_wmin][1:,]); coef_2 = beta_v(x[Omega_wmin]);
    
    # Update row
    row = np.concatenate((row, 0, 0, np.tile(Omega_wmin[0][1:], 3)), axis=None)
    
    column_wmin = np.tile(Omega_wmin[0][1:], 3)
    
    # Adjust indexes of column_wmin
    column_wmin[:len(Omega_wmin[0][1:])] -= N_1
    column_wmin[2*len(Omega_wmin[0][1:]):] += N_1
    
    #Update column
    column = np.concatenate((column, index(0,1), 0, column_wmin), axis=None)
    
    #Update data
    data = np.concatenate((data, coef_2[0], - coef_2[0], coef_1, -(coef_1 + coef_2[1:]), coef_2[1:]), axis=None)
    
    " Lower boundary, v = 0"
    # Update row
    row = np.concatenate((row, np.tile(Omega_vmin[0], 2)), axis=None)
    
    column_vmin = np.tile(Omega_vmin[0], 2)
    
    # Adjust indexes of column_wmin
    column_vmin[:len(Omega_vmin[0])] += N_1

    #Update column
    column = np.concatenate((column, column_vmin), axis=None)
    
    coef_2 = np.full(len(Omega_vmin[0]), coef_2[0])
    
    #Update data
    data = np.concatenate((data, coef_2, - coef_2), axis=None)
    
    " Interior points "
    # We first handel the drift term in the v-direction
    # Update row
    row = np.concatenate((row, np.tile(Omega_in[0], 3)), axis=None)
    
    column_in = np.tile(Omega_in[0], 3)
    
    # Adjust indexes of column_in
    column_in[:len(Omega_in[0])] -= N_1
    column_in[2*len(Omega_in[0]):] += N_1
    
    #Update column
    column = np.concatenate((column, column_in), axis=None)
    
    # Backward/forward differencing 
    coef_1 = -1 * (kappa*(theta - x[Omega_in][:,1]) < 0) * kappa*(theta - x[Omega_in][:,1])/h
    coef_2 = (kappa*(theta - x[Omega_in][:,1]) >= 0) * kappa*(theta - x[Omega_in][:,1])/h
    
    #Update data
    data = np.concatenate((data, coef_1, - (coef_1 + coef_2), coef_2), axis=None)
    
    # Calculate off-grid points
    angles = phi(controls[Omega_in], x[Omega_in])
    
    Psi_m = Psi(x[Omega_in[0]], angles)
    
    " No special treatment (NST) "
    coef_1 = a(controls[Omega_in_upper], x[Omega_in_upper], angles[Omega_in_NST])
    coef_2 = b(controls[Omega_in_upper], x[Omega_in_upper], angles[Omega_in_NST])
    
    #Update row
    row = np.concatenate((row, np.tile(Omega_in_upper[0], 20)), axis=None)
    
    for m in range(4):
        # We only include element that are not above the upper boundaries
        adjust = np.where(v_max < Psi_m[m][:,1][Omega_in_NST])
        
        Psi_m[m][:,1][Omega_in_NST[0][adjust]] = v_max
        
        omega_11, omega_12, omega_21, omega_22, index_left_w, index_left_v = improved_linear_interpolation(Psi_m[2][Omega_in_NST[0]], tau)
        
        if m < 2:
            #Update data
            data = np.concatenate((data, coef_1/h * omega_11, coef_1/h * omega_12,
                                    coef_1/h * omega_21, coef_1/h * omega_22, -coef_1/h), axis=None) 
        else:
            data = np.concatenate((data, coef_2/h * omega_11, coef_2/h * omega_12,
                                    coef_2/h * omega_21, coef_2/h * omega_22, -coef_2/h), axis=None)
            
        
        #Update column
        column = np.concatenate((column, index(index_left_w,index_left_v), index(index_left_w,index_left_v + 1),
                                  index(index_left_w + 1,index_left_v), index(index_left_w + 1,index_left_v + 1), Omega_in_upper[0]), axis=None) 
    
    
    " Special treatment lower (STL), i.e. avoid points below v = 0 "
    # Update row
    row = np.concatenate((row, np.tile(Omega_in_lower[0], 20)), axis=None)
    
    # Rotation matrix
    R = [np.array([np.cos(angles[Omega_in_STL]), np.sin(angles[Omega_in_STL])]).T, np.array([-np.sin(angles[Omega_in_STL]), np.cos(angles[Omega_in_STL])]).T]
    
    for m in [0, 2]:
        """ Algorithm 4.2 """
        h_right = np.full(len(Omega_in_STL[0]), h_sqrt)
        x_right = Psi_m[m][Omega_in_STL]
        
        below_right = np.where(x_right[:,1] < 0)
        
        # Shrink stencil
        h_right[below_right] = h
        x_right[below_right] = x[Omega_in_lower][below_right] + h * R[m//2][below_right]
        
        
        h_left = np.full(len(Omega_in_STL[0]), h_sqrt)
        x_left = Psi_m[m+1][Omega_in_STL]
        
        below_left = np.where(x_left[:,1] < 0)
        
        # Shrink stencil
        h_left[below_left] = h
        x_left[below_left] = x[Omega_in_lower][below_left] - h * R[m//2][below_left]
       
        coef_1 = 2/(h_left + h_right); coef_2 = [1/h_right, 1/h_left]
        
        for count, elmement in enumerate([x_right, x_left]):
            
            omega_11, omega_12, omega_21, omega_22, index_left_w, index_left_v = improved_linear_interpolation(elmement, tau)
            
            #Update data
            data = np.concatenate((data, coef_1 * coef_2[count]  * omega_11, coef_1 * coef_2[count] * omega_12,
                                    coef_1 * coef_2[count] * omega_21, coef_1 * coef_2[count] * omega_22, - coef_1 * coef_2[count]), axis=None)
            #Update column
            column = np.concatenate((column, index(index_left_w,index_left_v), index(index_left_w,index_left_v + 1),
                                      index(index_left_w + 1,index_left_v), index(index_left_w + 1,index_left_v + 1), Omega_in_lower[0]), axis=None)
    
    # Construct matrix using coo format and convert to CSR format, better format for arithmetic and matrix vector operations
    Lnp1 = coo_matrix((data, (row,column)),shape=(N,N)).tocsr()
    
    return Lnp1

# terminal_values = ((x[:,0]-eta/2)**2)
# opt_controls = np.ones_like(terminal_values)

# hej = L(opt_controls, x, dtau)

# omega_11, omega_12, omega_21, omega_22, index_left_w, index_left_v = improved_linear_interpolation(hej, dtau)

# We make an algorithm to preform the linear search for the optimal controls in the policy iteration 
def find_optima1_controls(W, U, tau):
    # Initialize minimum values
    minvalue = np.tile(np.inf, W.size)
    
    # Allocate storage for optimal controls
    opt_controls = np.zeros_like(W)
    
    for control in PIs: # Find the optimal control
        controls_guess = np.full(W.size, control)
        
        Lnp1 = L(controls_guess, x, tau)
        PHInp1 = PHI(controls_guess, x, U, tau)
        # Gnp1 = G(controls_guess, x, tau)
        
        value = PHInp1 + dtau * Lnp1.dot(W)
        # PHInp1 + dtau * (Lnp1.dot(W) + Gnp1)
        
        indices = np.where(value < minvalue)
        
        #Update
        minvalue[indices] = value[indices]
        opt_controls[indices] = controls_guess[indices]
        
    return opt_controls

I = identity(N)

# Terminal condition
terminal_values = ((x[:,0]-eta/2)**2)

# Allocate storage for boundary condtions
Fn = np.zeros_like(terminal_values)
Fnp1 = np.zeros_like(terminal_values)


timesteps = np.linspace( 0.0, T, N_tau + 1)[1:] # drop first item which is tau=0.0

U = terminal_values

# Allocate storage for controls
save_controls = np.zeros_like(timesteps)

# Find location in grid
index_left_w = np.searchsorted(w, w_0, side="right") - 1 
index_left_v = np.searchsorted(v, v_0, side="right") - 1

w_left = w[index_left_w]; w_right = w[index_left_w + 1]
v_left = v[index_left_v]; v_right = v[index_left_v + 1]

omega_11 = (w_right - w_0)*(v_right - v_0)/(dw * dv)
omega_12 = (w_right - w_0)*(v_0 - v_left)/(dw * dv)
omega_21 = (w_0 - w_left)*(v_right - v_0) /(dw * dv)
omega_22 = (w_0 - w_left)*(v_0-v_left) /(dw * dv)


for count, tau in enumerate(timesteps):
    
    W = U.copy()
    # Update boundary condition
    Fn[Omega_wmax] = BC(tau - dtau, w_max)
    Fnp1[Omega_wmax] = BC(tau, w_max)
    
    # Boundary condition
    Boundary_condition = Fnp1 - Fn
    
    """ Algorithm 4.1: Policy iteration """
    while True:
        # Linear search
        opt_controls = find_optima1_controls(W, U, tau)
        
        Lnp1 = L(opt_controls, x, tau)
        M = I - dtau * Lnp1
        
        
        RHS = PHI(opt_controls, x, U, tau) + Boundary_condition
        # PHI(opt_controls, x, U, tau) + dtau * G(opt_controls, x, tau) + Boundary_condition
        
        Wnew = spsolve(M, RHS) 
        # Wnew, check = bicgstab(M, RHS)
        # print(check)
       
        scale = np.maximum(np.abs(Wnew), np.ones_like(Wnew))
        residuals = np.abs(Wnew - W)/scale
        print(np.max(residuals))
        # print(np.max(residuals))
        if np.all(residuals < tol):
            U = Wnew
            break
        
        else:
            W = Wnew
    
    save_controls[count] = (omega_11 * opt_controls[index(index_left_w, index_left_v)] + omega_12 * opt_controls[index(index_left_w, index_left_v+1)]
                            + omega_21 * opt_controls[index(index_left_w+1, index_left_v)] + omega_22 * opt_controls[index(index_left_w+1, index_left_v+1)])
    
    print("Process: {0:.00%}".format(tau/T))
    
   
    
    
    







