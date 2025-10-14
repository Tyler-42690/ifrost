'''
    Utility functions for 3D and 1D electromagnetic induction (EMI) modeling.
    Includes functions for forward modeling, solving Maxwell's equations, and performing
    Hankel transforms using the modified W-transform method.
'''
import scipy.special as sp
import scipy.integrate as spi
import numpy as np
from numba import njit

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
                              x_min=6*np.pi, num_points=200, max_iterations=100, tolerance=1e-9):
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


# -----------------------
# Simpson rule for values y(x) with uniform spacing (Numba)
# -----------------------
@njit
def simpson_rule_numba_from_values(y, x):
    """
    Simpson's rule using precomputed y values and x grid (1D arrays).
    Ensures odd number of points by dropping last point if necessary.
    """
    n = len(x)
    if n < 2:
        return 0.0 + 0.0j
    # ensure odd number of points -> even number of subintervals
    if (n - 1) % 2 == 1:
        # drop last point
        n -= 1
    h = (x[n-1] - x[0]) / (n - 1)
    s = y[0] + y[n-1]
    # sum odd indices
    for i in range(1, n-1, 2):
        s += 4.0 * y[i]
    # sum even indices
    for i in range(2, n-2, 2):
        s += 2.0 * y[i]
    return s * h / 3.0

# -----------------------
# Numba-compatible TE reflection coefficient (already provided)
# -----------------------
@njit
def rte_function_numba(x, angfreq, permittivity, permeability, conductivity, layer_height):
    """
    Numba-compatible TE reflection coefficient for layered earth.
    x : 1D array of k = x/rho samples (but you can pass any x array)
    angfreq : scalar
    returns r_te (1D complex array same length as x)
    """
    mu0 = 4 * np.pi * 1e-7
    eps0 = 8.854e-12
    Nx = len(x)
    Nlayer = len(permeability)
    r_te = np.empty(Nx, dtype=np.complex128)

    for i in range(Nx):
        xi = x[i]
        k0_sq = (angfreq**2) * mu0 * eps0
        u0 = np.sqrt(xi**2 - k0_sq + 0j)
        y0 = u0 / (1j * angfreq * mu0)

        # Bottom layer
        n = Nlayer - 1
        kn_sq = (angfreq**2 * permittivity[n] * permeability[n]
                 - 1j * angfreq * conductivity[n] * permeability[n])
        un = np.sqrt(xi**2 - kn_sq + 0j)
        yn = un / (1j * angfreq * permeability[n])

        y_hat = yn

        # Upward recursion
        for nl in range(Nlayer - 2, -1, -1):
            kn_sq = (angfreq**2 * permittivity[nl] * permeability[nl]
                     - 1j * angfreq * conductivity[nl] * permeability[nl])
            un = np.sqrt(xi**2 - kn_sq + 0j)
            yn = un / (1j * angfreq * permeability[nl])

            e = np.exp(-2 * un * layer_height[nl])

            num = y_hat * (1.0 + e) + yn * (1.0 - e)
            den = yn * (1.0 + e) + y_hat * (1.0 - e)
            y_hat = yn * (num / den)

        r_te[i] = (y0 - y_hat) / (y0 + y_hat)

    return r_te

# -----------------------
# Integrand (top-level) for a given x scalar and frequency w
# -----------------------
@njit
def integrand_numba_scalar(x, w, rho, htx, zrx,
                           permittivity, permeability, conductivity, layer_height):
    """
    scalar integrand f(x) = J0(x) * g(x) for a single x and frequency w.
    Note: here x is the Bessel-domain variable (like original code).
    """
    mu0 = 4 * np.pi * 1e-7
    eps0 = 8.854e-12

    # compute u0 for free space using k0
    k0_sq = w**2 * mu0 * eps0
    # argument for u0 uses (x/rho)
    xrho = x / rho
    u0 = np.sqrt(xrho**2 - k0_sq + 0j)

    # compute r_te for this single xrho: call rte_function_numba with array of one element
    k_arr = np.empty(1, dtype=np.float64)
    k_arr[0] = xrho
    r_te_arr = rte_function_numba(k_arr, w, permittivity, permeability, conductivity, layer_height)
    r_te = r_te_arr[0]

    # j0: use scipy.special.j0 on Python side is not allowed in njit; but many Numba builds support sp.j0.
    # We call np.where fallback if j0 unavailable — here we attempt to call scipy.special.j0 via Python function call,
    # but inside njit it may or may not work depending on Numba version. If that fails, precompute outside.
    j0x = sp.jv(0,x)  # many setups allow this in njit; if your Numba complains, precompute j0 table outside

    return j0x * ( (x / rho)**3 ) / (u0 * rho) * r_te * np.exp(u0 * (zrx - htx))

# -----------------------
# Numba W-transform that takes frequency scalar w and all params (no nested functions)
# -----------------------
@njit
def mw_j0_integral_numba_scalar_w(w, rho, htx, zrx,
                                  permittivity, permeability, conductivity, layer_height,
                                  x_min=6*np.pi, num_points=200, max_iterations=100, tolerance=1e-9):
    """
    Compute integral for a single frequency w using the modified W-transform.
    Uses Simpson sampling for segments. No nested functions; Numba-friendly.
    """

    # --- initial serial integral 0 -> x_min using Simpson sampling ---
    # create x grid for [0, x_min]
    N0 = num_points
    if N0 % 2 == 0:
        N0 += 1
    x0 = np.linspace(0.0, x_min, N0)
    fvals0 = np.empty(N0, dtype=np.complex128)
    for i in range(N0):
        fvals0[i] = integrand_numba_scalar(x0[i], w, rho, htx, zrx,
                                          permittivity, permeability, conductivity, layer_height)
    integral_initial = simpson_rule_numba_from_values(fvals0, x0)

    # --- prepare approximate Bessel zeros (spacing ~ pi) ---
    max_iter = max_iterations
    x_zeros = np.empty(max_iter + 3, dtype=np.float64)
    x_zeros[0] = x_min
    for i in range(max_iter + 2):
        x_zeros[i + 1] = x_zeros[i] + np.pi

    # --- First three segments using Simpson sampling ---
    # segment x_min -> x_zeros[0]
    Nseg = num_points if (num_points % 2 == 1) else num_points + 1
    x_seg = np.linspace(x_min, x_zeros[0], Nseg)
    fseg = np.empty(Nseg, dtype=np.complex128)
    for i in range(Nseg):
        fseg[i] = integrand_numba_scalar(x_seg[i], w, rho, htx, zrx,
                                         permittivity, permeability, conductivity, layer_height)
    f0 = simpson_rule_numba_from_values(fseg, x_seg)

    # psi_0: x_zeros[0] -> x_zeros[1]
    x_seg = np.linspace(x_zeros[0], x_zeros[1], Nseg)
    for i in range(Nseg):
        fseg[i] = integrand_numba_scalar(x_seg[i], w, rho, htx, zrx,
                                         permittivity, permeability, conductivity, layer_height)
    psi_0 = simpson_rule_numba_from_values(fseg, x_seg)

    # psi_1: x_zeros[1] -> x_zeros[2]
    x_seg = np.linspace(x_zeros[1], x_zeros[2], Nseg)
    for i in range(Nseg):
        fseg[i] = integrand_numba_scalar(x_seg[i], w, rho, htx, zrx,
                                         permittivity, permeability, conductivity, layer_height)
    psi_1 = simpson_rule_numba_from_values(fseg, x_seg)

    # --- W-transform initial algebra ---
    # protect against tiny psi values by adding tiny eps if needed
    eps = 1e-30
    psi_0_safe = psi_0 if np.abs(psi_0) > eps else psi_0 + eps
    psi_1_safe = psi_1 if np.abs(psi_1) > eps else psi_1 + eps

    m_0 = f0 / psi_0_safe
    n_0 = 1.0 / psi_0_safe
    f1 = f0 + psi_0

    m_1 = np.zeros(2, dtype=np.complex128)
    n_1 = np.zeros(2, dtype=np.complex128)
    m_1[1] = f1 / psi_1_safe
    n_1[1] = 1.0 / psi_1_safe
    denom01 = (1.0 / (x_zeros[0] + 1e-30) - 1.0 / (x_zeros[1] + 1e-30))
    m_1[0] = (m_0 - m_1[1]) / denom01
    n_1[0] = (n_0 - n_1[1]) / denom01

    integral_value = m_1[0] / n_1[0]

    # --- W-transform iterative loop ---
    iteration = 2
    while True:
        # compute next psi segment
        if iteration + 1 >= x_zeros.shape[0]:
            break
        x_seg = np.linspace(x_zeros[iteration], x_zeros[iteration + 1], Nseg)
        for i in range(Nseg):
            fseg[i] = integrand_numba_scalar(x_seg[i], w, rho, htx, zrx,
                                             permittivity, permeability, conductivity, layer_height)
        psi_next = simpson_rule_numba_from_values(fseg, x_seg)

        # shift previous values
        psi_0_prev = psi_1
        m_0_prev = m_1.copy()
        n_0_prev = n_1.copy()
        f0_prev = f1

        psi_1 = psi_next
        f1 = f0_prev + psi_0_prev

        # safe division
        psi_1_safe = psi_1 if np.abs(psi_1) > eps else psi_1 + eps

        # build m_1, n_1 for current iteration
        m_1 = np.zeros(iteration + 1, dtype=np.complex128)
        n_1 = np.zeros(iteration + 1, dtype=np.complex128)
        m_1[iteration] = f1 / psi_1_safe
        n_1[iteration] = 1.0 / psi_1_safe

        # backward recursion to compute other m_1/n_1 entries
        for k in range(iteration - 1, -1, -1):
            weight = 1.0 / (x_zeros[k] + 1e-30) - 1.0 / (x_zeros[iteration] + 1e-30)
            # m_0_prev and n_0_prev have length iteration
            # guard indexing: if k < m_0_prev.shape[0], use their value else fallback 0
            a = m_0_prev[k] if k < m_0_prev.shape[0] else 0.0 + 0.0j
            b = n_0_prev[k] if k < n_0_prev.shape[0] else 0.0 + 0.0j
            m_1[k] = (a - m_1[k+1]) / weight
            n_1[k] = (b - n_1[k+1]) / weight

        integral_new = m_1[0] / n_1[0]

        # convergence check
        if np.abs((integral_new - integral_value) / (integral_value + 1e-30)) < tolerance:
            integral_value = integral_new
            break

        integral_value = integral_new
        iteration += 1
        if iteration > max_iterations:
            break

    return integral_value + integral_initial

# END OF UTILS