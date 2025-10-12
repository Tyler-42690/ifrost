'''
    Utility functions for 3D and 1D electromagnetic induction (EMI) modeling.
    Includes functions for forward modeling, solving Maxwell's equations, and performing
    Hankel transforms using the modified W-transform method.
'''
import scipy.special as sp
import scipy.integrate as spi
import numpy as np


# 3D Forward Modeling Functions

def forward_fd_matrix(s, ge, dl_factor_cell, grid3d):
    '''
        To be implemented. Creates the forward finite difference matrix for solving the 3D EMI 
        Forward Problem. Needs a grid type from the electric field and grid properties (i.e. size) 
    '''
    
    return s, ge, dl_factor_cell, grid3d #TEMPORARY, REMOVE WHEN DONE CODING METHOD IN np

def create_curls(ge, d1_factor_cell, grid3d):
    '''
        To be implemented. Creates the curl operator that links the electric and magnetic fields
        via a staggered grid setup. In this case, a formulation that takes the curl of the E
        field to obtain the H field is used to find the secondary magnetic response.
    '''

    return ge, d1_factor_cell, grid3d #TEMPORARY, REMOVE WHEN DONE CODING METHOD IN np

def create_divs(ge, d1_factor_cell, grid3d):
    '''
        To be implemented. Creates the divergence operator for E and H fields. 
    '''

    return ge, d1_factor_cell, grid3d #TEMPORARY, REMOVE WHEN DONE CODING METHOD IN np

def create_masks(ge, grid3d):
    '''
        Generates indices to mask to avoid NaN or Inf values going to 0 when multiplying matrices.
    '''

    return ge, grid3d #TEMPORARY, REMOVE WHEN DONE CODING METHOD IN np


def solve_eq_direct(eqtype, pml, omega, eps_cell, mu_cell, s_factor_cell, J_cell, M_cell, grid3d):
    '''
        Directly solves Maxwell's equations in 3D, though at a potentially too significant cost on 
        local devices. Returns the secondary magnetic field after solving with respect to multiple frequencies.
    '''

    return eqtype, pml, omega, eps_cell, mu_cell, s_factor_cell, J_cell, M_cell, grid3d #TEMPORARY, REMOVE WHEN DONE CODING METHOD IN np

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
    n_0 = np.ceil(x_min / np.pi - 5/4).to(np.int32) #Lowest zero approx (Larger than xMin)
    x_zero = n_0 * np.pi + 5*np.pi/4
    x_zero = besselj1_zero(x_zero) #Find zero of J_1(x) close to xZero

    #Iteratively find all zeros of J_1(x) larger than xZero

    num_zeros = np.floor((x_max - x_zero) / np.pi).to(np.int32)
    z_s = np.zeros(num_zeros,1)
    z_s[0] = x_zero
    for i in range(1, num_zeros):
        x_zero = besselj1_zero(x_zero+np.pi)
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
        if np.abs(x_zero_new - x_zero) <= eps:
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
    n_0 = np.ceil(x_min / np.pi - 3/4).to(np.int32) #Lowest zero approx (Larger than xMin)
    x_zero = n_0 * np.pi + 3*np.pi/4
    x_zero = besselj0_zero(x_zero) #Find zero of J_0(x) close to xZero

    #Iteratively find all zeros of J_0(x) larger than xZero

    num_zeros = np.floor((x_max - x_zero) / np.pi).to(np.int32)
    z_s = np.zeros(num_zeros,1)
    z_s[0] = x_zero
    for i in range(1, num_zeros):
        x_zero = besselj0_zero(x_zero+np.pi)
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
        if np.abs(x_zero_new - x_zero) <= eps:
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

def simpson_vectorized(y, x):
    """
    Vectorized Simpson's rule along last axis.
    y: shape (num_freqs, num_points)
    x: shape (num_points,)
    Returns: array of integrals shape (num_freqs,)
    """
    n = x.size
    if n % 2 == 0:
        n -= 1
        x = x[:n]
        y = y[:, :n]

    h = (x[-1] - x[0]) / (n - 1)
    w = np.ones(n)
    w[1:-1:2] = 4
    w[2:-2:2] = 2
    return (h/3) * np.sum(y * w[None, :], axis=-1)

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
    tolerance = 1e-9

    # -------------------------------------------------------------------------
    # Generate zeros of J0 using custom root finder
    x_zeros = np.zeros(max_iterations + 3)
    x_zeros[0] = besselj0_zero(x_min + np.pi)
    for i in range(max_iterations + 2):
        x_zeros[i + 1] = besselj0_zero(x_zeros[i] + np.pi)
    

    # -------------------------------------------------------------------------
    # Integral from 0 to x_min
   
    integral_inital, _ = spi.quad(f, 0, x_min, epsabs=tolerance, complex_func=True)

    # -------------------------------------------------------------------------
    # Modified W-transform method
    f0 = simpson_rule(f, x_min, x_zeros[0], num_points)
    psi_0 = simpson_rule(f, x_zeros[0], x_zeros[1], num_points)
    psi_1 = simpson_rule(f, x_zeros[1], x_zeros[2], num_points)

    m_0 = f0 / psi_0
    n_0 = 1 / psi_0
    f1 = f0 + psi_0

    m_1 = np.zeros(2,dtype=np.complex128)
    n_1 = np.zeros(2,dtype=np.complex128)
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

        m_1 = np.zeros(iteration,dtype=np.complex128)
        n_1 = np.zeros(iteration,dtype=np.complex128)
        m_1[iteration - 1] = f1 / psi_1
        n_1[iteration - 1] = 1 / psi_1

        for k in range(iteration - 2, -1, -1):
            weight = 1 / x_zeros[k] - 1 / x_zeros[iteration]
            m_1[k] = (m_0[k] - m_1[k + 1]) / weight
            n_1[k] = (n_0[k] - n_1[k + 1]) / weight

        integral_new = m_1[0] / n_1[0]

        if np.abs((integral_new - integral_value) / integral_value) < tolerance or iteration >= max_iterations:
            stop = True

        integral_value = integral_new

    return integral_value + integral_inital

def mw_j0_integral_vectorized(f_callable, angfreqs,
                              x_min=6*np.pi, num_points=200, max_iterations=100, tolerance=1e-8):
    """
    Vectorized Modified W-transform integration of f(x) = J0(x) * g(x) for multiple frequencies.
    
    Parameters
    ----------
    f_callable : callable
        Function of (x, idx) returning complex integrand for frequency idx
    angfreqs : array-like
        Angular frequencies, shape (num_freqs,)
    rho, zrx, htx : float
        Geometry parameters
    permittivity, permeability, conductivity, layer_height : arrays
        Layer properties
    x_min : float
        Start of W-transform
    num_points : int
        Number of Simpson points per segment
    max_iterations : int
        Max W-transform iterations
    tolerance : float
        Relative tolerance for convergence
    
    Returns
    -------
    integral_value : array
        Computed integral for each frequency, shape (num_freqs,)
    """
    num_freqs = len(angfreqs)
    # Approximate zeros of J0
    x_zeros = np.zeros(max_iterations + 3)
    x_zeros[0] = x_min
    for i in range(max_iterations + 2):
        x_zeros[i+1] = x_zeros[i] + np.pi

    # -------------------------------------------------------------------------
    # Helper: vectorized Simpson integration
    #

    # -------------------------------------------------------------------------
    # Compute initial integral from 0 to x_min using quad (per frequency)
    integral_initial = np.zeros(num_freqs, dtype=np.complex128)
    for i in range(num_freqs):
        integral_initial[i], _ = spi.quad(lambda x: f_callable(x, i), 0, x_min,
                                         epsabs=tolerance, complex_func=True)

    # -------------------------------------------------------------------------
    # Compute segment integrals using vectorized Simpson
    def compute_segment(x0, x1):
        x_vals = np.linspace(x0, x1, num_points)
        y_vals = np.array([f_callable(x_vals, i) for i in range(num_freqs)])
        return simpson_vectorized(y_vals, x_vals)

    # Initial segments
    f0 = compute_segment(x_min, x_zeros[0])
    psi_0 = compute_segment(x_zeros[0], x_zeros[1])
    psi_1 = compute_segment(x_zeros[1], x_zeros[2])

    m_0 = f0 / psi_0
    n_0 = 1 / psi_0
    f1 = f0 + psi_0

    m_1 = np.zeros((num_freqs, 2), dtype=np.complex128)
    n_1 = np.zeros((num_freqs, 2), dtype=np.complex128)
    m_1[:,1] = f1 / psi_1
    n_1[:,1] = 1 / psi_1
    m_1[:,0] = (m_0 - m_1[:,1]) / (1 / x_zeros[0] - 1 / x_zeros[1])
    n_1[:,0] = (n_0 - n_1[:,1]) / (1 / x_zeros[0] - 1 / x_zeros[1])

    integral_value = m_1[:,0] / n_1[:,0]

    # -------------------------------------------------------------------------
    # W-transform iteration
    stop = np.zeros(num_freqs, dtype=bool)
    iteration = 2

    while not np.all(stop) and iteration < max_iterations:
        iteration += 1
        psi_0, m_0, n_0, f0 = psi_1, m_1.copy(), n_1.copy(), f1.copy()
        psi_1 = compute_segment(x_zeros[iteration], x_zeros[iteration+1])
        f1 = f0 + psi_0  # no indexing needed, shape (num_freqs,)

        m_1 = np.zeros((num_freqs, iteration), dtype=np.complex128)
        n_1 = np.zeros((num_freqs, iteration), dtype=np.complex128)
        m_1[:, iteration-1] = f1 / psi_1
        n_1[:, iteration-1] = 1 / psi_1

        for k in range(iteration-2, -1, -1):
            weight = 1 / x_zeros[k] - 1 / x_zeros[iteration]
            m_1[:,k] = (m_0[:,k] - m_1[:,k+1]) / weight
            n_1[:,k] = (n_0[:,k] - n_1[:,k+1]) / weight

        integral_new = m_1[:,0] / n_1[:,0]
        rel_error = np.abs((integral_new - integral_value)/integral_value)
        stop = rel_error < tolerance
        integral_value = integral_new

    return integral_value + integral_initial
# END OF UTILS