import torch
import scipy.special as sp

torch.set_default_dtype(torch.float64)  # Set precision to 64-bit
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# 3D Forward Modeling Functions

def forward_fd_matrix(s, ge, dl_factor_cell, grid3d):
    '''
        To be implemented. Creates the forward finite difference matrix for solving the 3D EMI 
        Forward Problem. Needs a grid type from the electric field and grid properties (i.e. size) 
    '''
    
    return s, ge, dl_factor_cell, grid3d #TEMPORARY, REMOVE WHEN DONE CODING METHOD IN TORCH

def create_curls(ge, d1_factor_cell, grid3d):
    '''
        To be implemented. Creates the curl operator that links the electric and magnetic fields
        via a staggered grid setup. In this case, a formulation that takes the curl of the E
        field to obtain the H field is used to find the secondary magnetic response.
    '''

    return ge, d1_factor_cell, grid3d #TEMPORARY, REMOVE WHEN DONE CODING METHOD IN TORCH

def create_divs(ge, d1_factor_cell, grid3d):
    '''
        To be implemented. Creates the divergence operator for E and H fields. 
    '''

    return ge, d1_factor_cell, grid3d #TEMPORARY, REMOVE WHEN DONE CODING METHOD IN TORCH

def create_masks(ge, grid3d):
    '''
        Generates indices to mask to avoid NaN or Inf values going to 0 when multiplying matrices.
    '''

    return ge, grid3d #TEMPORARY, REMOVE WHEN DONE CODING METHOD IN TORCH


def solve_eq_direct(eqtype, pml, omega, eps_cell, mu_cell, s_factor_cell, J_cell, M_cell, grid3d):
    '''
        Directly solves Maxwell's equations in 3D, though at a potentially too significant cost on 
        local devices. Returns the secondary magnetic field after solving with respect to multiple frequencies.
    '''

    return eqtype, pml, omega, eps_cell, mu_cell, s_factor_cell, J_cell, M_cell, grid3d #TEMPORARY, REMOVE WHEN DONE CODING METHOD IN TORCH

def solve_eq_iterative(maxit, tol, F0type, eqtype, pml, omega, eps_cell, mu_cell, s_factor_cell, J_cell, M_cell, grid3d):
    '''
        Iteratively solves Maxwell's equations in 3D, at a much lower cost due to iterative techniques. 
        Returns the secondary magnetic field after solving with respect to multiple frequencies.
    '''

    return maxit, tol, F0type, eqtype, pml, omega, eps_cell, mu_cell, s_factor_cell, J_cell, M_cell, grid3d

#1D Forward Modeling Functions

def besselj1_zeros(x_min, x_max):
    '''
        compute all zeros of the Bessel function J_1(x) between xMin and xMax
        Idea: use Newton's method to compute all zeros, under assumption that the
        zeros are approximately pi unit from each other. This is true when xMin
        is large enough based on the assymptotic expansion:
        J_1(x) \approx sqrt(2/pi/x)*cos(x - 3*pi/4)
    '''
    n_0 = torch.ceil(x_min / torch.pi - 5/4).to(torch.int32) #Lowest zero approx (Larger than xMin)
    x_zero = n_0 * torch.pi + 5*torch.pi/4
    x_zero = besselj1_zero(x_zero) #Find zero of J_1(x) close to xZero

    #Iteratively find all zeros of J_1(x) larger than xZero

    num_zeros = torch.floor((x_max - x_zero) / torch.pi).to(torch.int32)
    z_s = torch.zeros(num_zeros,1, dtype=torch.float64, device=x_min.device)
    z_s[0] = x_zero
    for i in range(1, num_zeros):
        x_zero = besselj1_zero(x_zero+torch.pi)
        z_s[i] = x_zero
    return z_s 

def besselj1_zero(x_zero):
    '''
        Use Newton's method to find a zero of the Bessel function J_1(x) close to x0
    '''
    eps = 1e-15
    stop = 0
    while not stop:
        x_zero_new = x_zero + sp.jv(1,x_zero)/(sp.jv(2, x_zero) - sp.jv(1,x_zero)/x_zero)
        if torch.abs(x_zero_new - x_zero) <= eps:
            stop = 1
        x_zero = x_zero_new
    return x_zero

def besselj0_zeros(x_min, x_max):
    '''
        compute all zeros of the Bessel function J_0(x) between xMin and xMax
        Idea: use Newton's method to compute all zeros, under assumption that the
        zeros are approximately pi unit from each other. This is true when xMin
        is large enough based on the assymptotic expansion:
        J_0(x) \approx sqrt(2/pi/x)*cos(x - pi/4)
    '''
    n_0 = torch.ceil(x_min / torch.pi - 3/4).to(torch.int32) #Lowest zero approx (Larger than xMin)
    x_zero = n_0 * torch.pi + 3*torch.pi/4
    x_zero = besselj0_zero(x_zero) #Find zero of J_0(x) close to xZero

    #Iteratively find all zeros of J_0(x) larger than xZero

    num_zeros = torch.floor((x_max - x_zero) / torch.pi).to(torch.int32)
    z_s = torch.zeros(num_zeros,1, dtype=torch.float64, device=x_min.device)
    z_s[0] = x_zero
    for i in range(1, num_zeros):
        x_zero = besselj0_zero(x_zero+torch.pi)
        z_s[i] = x_zero
    return z_s

def besselj0_zero(x_zero):
    '''
        Use Newton's method to find a zero of the Bessel function J_0(x) close to x0
    '''
    eps = 1e-15
    stop = 0
    while not stop:
        x_zero_new = x_zero + sp.jv(0,x_zero)/(sp.jv(1,x_zero))
        if torch.abs(x_zero_new - x_zero) <= eps:
            stop = 1
        x_zero = x_zero_new
    return x_zero

