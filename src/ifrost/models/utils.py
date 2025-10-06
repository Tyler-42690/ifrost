import torch
import scipy.special as sp
import scipy.integrate as spi
import numpy as np

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

# Modified W Transform Functions
def simpson_rule(f, a, b, num_points):
    """
    Simpson's rule integration using num_points equally spaced samples between a and b.
    num_points should be even for classical Simpson's rule (like MATLAB).
    """
    if num_points % 2 == 1:
        num_points += 1  # ensure even number of intervals
    x = np.linspace(a, b, num_points + 1)
    y = np.array([f(xi) for xi in x])
    return spi.simpson(y, x)

def mw_j1_integral(f):
    """
    Compute the integral of f(x) = J1(x) * g(x) from 0 to infinity
    using the modified W-transform method of Sidi (1988).

    This is used in the Hankel transform for the magnetic field of a circular loop source.

    Parameters
    ----------
    f : callable
        Function handle for the integrand (e.g., f(x) = J1(x) * g(x))

    Returns
    -------
    integral_value : float
        Computed integral value
    """
    # =========================================================================
    num_points = 200        # number of Simpson points between zeros
    max_iterations = 100    # maximum iterations
    x_min = 5 * np.pi       # the integral from 0 to x_min is computed directly
    tolerance = 1e-14

    # -------------------------------------------------------------------------
    # Generate zeros of J1 using custom root finder
    x_zeros = np.zeros(max_iterations + 3)
    x_zeros[0] = besselj1_zero(x_min + np.pi)
    for i in range(max_iterations + 2):
        x_zeros[i + 1] = besselj1_zero(x_zeros[i] + np.pi)

    # -------------------------------------------------------------------------
    # Integral from 0 to x_min
    integral_initial, _ = spi.quad(f, 0, x_min, epsabs=tolerance)

    # -------------------------------------------------------------------------
    # Modified W-transform method
    f0 = simpson_rule(f, x_min, x_zeros[0], num_points)
    psi_0 = simpson_rule(f, x_zeros[0], x_zeros[1], num_points)
    psi_1 = simpson_rule(f, x_zeros[1], x_zeros[2], num_points)

    m_0 = f0 / psi_0
    n_0 = 1 / psi_0
    f1 = f0 + psi_0

    m_1 = np.zeros(2)
    n_1 = np.zeros(2)
    m_1[1] = f1 / psi_1
    n_1[1] = 1 / psi_1
    m_1[0] = (m_0 - m_1[1]) / (1 / x_zeros[0] - 1 / x_zeros[1])
    n_1[0] = (n_0 - n_1[1]) / (1 / x_zeros[0] - 1 / x_zeros[1])

    integral_value = m_1[0] / n_1[0]

    # -------------------------------------------------------------------------
    stop = False
    iteration = 2
    while not stop:
        iteration += 1
        psi_prev, m_0, n_0, f0 = psi_1, m_1, n_1, f1

        psi_1 = simpson_rule(f, x_zeros[iteration - 1], x_zeros[iteration], num_points)
        f1 = f0 + psi_prev

        m_1 = np.zeros(iteration)
        n_1 = np.zeros(iteration)
        m_1[iteration - 1] = f1 / psi_1
        n_1[iteration - 1] = 1 / psi_1

        for k in range(iteration - 2, -1, -1):
            weight = 1 / x_zeros[k] - 1 / x_zeros[iteration - 1]
            m_1[k] = (m_0[k] - m_1[k + 1]) / weight
            n_1[k] = (n_0[k] - n_1[k + 1]) / weight

        integral_new = m_1[0] / n_1[0]

        if abs((integral_new - integral_value) / integral_value) < tolerance or iteration >= max_iterations:
            stop = True

        integral_value = integral_new

    return integral_value + integral_initial

def mw_j0_integral(f):
    """
    Compute the integral of f(x) = J0(x) * g(x) from 0 to infinity
    using the modified W-transform method of Sidi (1988).

    Parameters
    ----------
    f : callable
        Function handle for the integrand f(x) = J0(x) * g(x)

    Returns
    -------
    integral_value : float
        Computed integral value
    """
    # =========================================================================
    num_points = 200        # number of Simpson points between zeros
    max_iterations = 100    # maximum iterations
    x_min = 6 * np.pi
    tolerance = 1e-8

    # -------------------------------------------------------------------------
    # Generate zeros of J0 using custom root finder
    x_zeros = np.zeros(max_iterations + 3)
    x_zeros[0] = besselj0_zero(x_min + np.pi)
    for i in range(max_iterations + 2):
        x_zeros[i + 1] = besselj0_zero(x_zeros[i] + np.pi)
    x_zeros[0] = besselj0_zero(x_min + np.pi)

    # -------------------------------------------------------------------------
    # Integral from 0 to x_min
    integral_inital, _ = spi.quad(f, 0, x_min, epsabs=tolerance)

    # -------------------------------------------------------------------------
    # Modified W-transform method
    f0 = simpson_rule(f, x_min, x_zeros[0], num_points)
    psi_0 = simpson_rule(f, x_zeros[0], x_zeros[1], num_points)
    psi_1 = simpson_rule(f, x_zeros[1], x_zeros[2], num_points)

    m_0 = f0 / psi_0
    n_0 = 1 / psi_0
    f1 = f0 + psi_0

    m_1 = np.zeros(2)
    n_1 = np.zeros(2)
    m_1[1] = f1 / psi_1
    n_1[1] = 1 / psi_1
    m_1[0] = (m_0 - m_1[1]) / (1 / x_zeros[0] - 1 / x_zeros[1])
    n_1[0] = (n_0 - n_1[1]) / (1 / x_zeros[0] - 1 / x_zeros[1])

    integral_value = m_1[0] / n_1[0]

    # -------------------------------------------------------------------------
    stop = False
    iteration = 2
    while not stop:
        iteration += 1
        psi_0, m_0, n_0, f0 = psi_1, m_1, n_1, f1

        psi_1 = simpson_rule(f, x_zeros[iteration], x_zeros[iteration + 1], num_points)
        f1 = f0 + psi_0

        m_1 = np.zeros(iteration)
        n_1 = np.zeros(iteration)
        m_1[iteration - 1] = f1 / psi_1
        n_1[iteration - 1] = 1 / psi_1

        for k in range(iteration - 2, -1, -1):
            weight = 1 / x_zeros[k] - 1 / x_zeros[iteration]
            m_1[k] = (m_0[k] - m_1[k + 1]) / weight
            n_1[k] = (n_0[k] - n_1[k + 1]) / weight

        integral_new = m_1[0] / n_1[0]

        if abs((integral_new - integral_value) / integral_value) < tolerance or iteration >= max_iterations:
            stop = True

        integral_value = integral_new

    return integral_value + integral_inital

# END OF UTILS