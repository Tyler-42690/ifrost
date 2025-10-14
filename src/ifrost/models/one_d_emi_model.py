import numpy as np
import scipy.special as sp
from joblib import Parallel, delayed
from numba import njit
from src.ifrost.models.utils import mw_j0_integral, mw_j1_integral, mw_j0_integral_vectorized, mw_j0_integral_numba_scalar_w

def rte_function(x, angfreq, permittivity, permeability, conductivity, layer_height):
    """
    Calculates the Transverse Electric (TE) reflection coefficient according to Ward & Hohmann (Eq. 4.19).

    Parameters
    ----------
    x : float or ndarray
        Wavenumber variable.
    angfreq : float
        Angular frequency (rad/s).
    permittivity : array_like
        Layer permittivity values.
    permeability : array_like
        Layer permeability values.
    conductivity : array_like
        Layer conductivity values.
    layer_height : array_like
        Thickness of each layer (last layer is semi-infinite).

    Returns
    -------
    r_transverse_electric : complex or ndarray
        Transverse Electric reflection coefficient.
    """

    # Number of layers
    layers = len(permeability)
    x = np.asarray(x, dtype=np.complex128)
    # Fundamental constants
    permeability_0 = 4 * np.pi * 1e-7   # Henries/meter
    permittivity_0 = 8.854e-12          # Farads/meter

    # Free-space parameters
    k0_sq = angfreq**2 * permittivity_0 * permeability_0
    u0 = np.sqrt(x**2 - k0_sq)
    z0 = 1j * angfreq * permeability_0
    y0 = u0 / z0  # Intrinsic admittance of free space

    # Bottom (deepest) layer parameters
    n = layers - 1
    kn_sq = angfreq**2 * permittivity[n] * permeability[n] - 1j * angfreq * conductivity[n] * permeability[n]
    zn = 1j * angfreq * permeability[n]
    un = np.sqrt(x**2 - kn_sq)
    yn = un / zn

    # Initialize Ŷ(layers) = Yn(layers)
    y_hat = yn

    # Loop upward from bottom layer to surface
    for n in range(layers - 2, -1, -1):
        kn_sq = angfreq**2 * permittivity[n] * permeability[n] - 1j * angfreq * conductivity[n] * permeability[n]
        zn = 1j * angfreq * permeability[n]
        un = np.sqrt(x**2 - kn_sq)
        yn = un / zn

        e = np.exp(-2 * un * layer_height[n])
        y_hat = yn * (y_hat * (1 + e) + yn * (1 - e)) / (yn * (1 + e) + y_hat * (1 - e))

    # Reflection coefficient for TE mode
    r_transverse_electric = (y0 - y_hat) / (y0 + y_hat)
    return r_transverse_electric

@njit
def rte_function_numba(x, angfreq, permittivity, permeability, conductivity, layer_height):
    """
    Numba-compatible TE reflection coefficient for layered earth.
    
    Parameters
    ----------
    x : ndarray (Nx,)
        Wavenumber samples divided by rho
    angfreq : float
        Single angular frequency
    permittivity : ndarray (Nlayer,)
    permeability : ndarray (Nlayer,)
    conductivity : ndarray (Nlayer,)
    layer_height : ndarray (Nlayer-1,)
    
    Returns
    -------
    r_te : ndarray (Nx,)
        TE reflection coefficient for all x at this frequency
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
        for n in range(Nlayer - 2, -1, -1):
            kn_sq = (angfreq**2 * permittivity[n] * permeability[n]
                     - 1j * angfreq * conductivity[n] * permeability[n])
            un = np.sqrt(xi**2 - kn_sq + 0j)
            yn = un / (1j * angfreq * permeability[n])
            
            e = np.exp(-2 * un * layer_height[n])
            
            num = y_hat * (1.0 + e) + yn * (1.0 - e)
            den = yn * (1.0 + e) + y_hat * (1.0 - e)
            y_hat = yn * (num / den)
        
        # TE reflection coefficient
        r_te[i] = (y0 - y_hat) / (y0 + y_hat)
    
    return r_te

# -------------------------------------------------------------------------
# Numba-compatible Simpson integration
@njit
def simpson_numba(y, x):
    """
    Simpson's rule for 1D arrays.
    """
    N = len(x)
    if N < 2:
        return 0.0 + 0.0j
    if N % 2 == 0:
        N -= 1  # make odd number of points
    h = (x[N-1] - x[0]) / (N-1)
    s = y[0] + y[N-1]
    for i in range(1, N-1, 2):
        s += 4.0 * y[i]
    for i in range(2, N-2, 2):
        s += 2.0 * y[i]
    return s * h / 3.0

# -------------------------------------------------------------------------
@njit
def integrand_numba(x, w, rho, htx, zrx, permittivity, permeability, conductivity, layer_height):
    mu0 = 4*np.pi*1e-7
    eps0 = 8.854e-12
    u0 = np.sqrt((x / rho)**2 - w**2 * mu0 * eps0 + 0j)
    r_te = rte_function_numba(np.array([x / rho]), w, permittivity, permeability, conductivity, layer_height)[0]
    return sp.jv(0, x) * ((x / rho)**3) / (u0 * rho) * r_te * np.exp(u0 * (zrx - htx))
# ------------------------ Numba JIT Forward Solver ------------------------
# -----------------------
# Final Numba JIT forward function (parallel over frequencies)
# -----------------------
@njit(parallel=True)
def forward_problem_mag_dipole_hz_numba(
    angfreqs, rho, mag_mom, htx, zrx,
    permittivity, permeability, conductivity, layer_height,
    x_min=6*np.pi, num_points=200, max_iterations=100
):
    """
    Parallel Numba implementation using the modified W-transform per frequency.
    """
    Nfreq = len(angfreqs)
    Hz = np.empty(Nfreq, dtype=np.complex128)

    for fi in range(Nfreq):
        w = angfreqs[fi]
        val = mw_j0_integral_numba_scalar_w(
            w, rho, htx, zrx,
            permittivity, permeability, conductivity, layer_height,
            x_min, num_points, max_iterations, 1e-9
        )
        Hz[fi] = mag_mom / (4.0 * np.pi) * val

    return Hz


# def rte_function_vectorized(x, angfreq, permittivity, permeability, conductivity, layer_height):
#     """
#     Vectorized Transverse Electric reflection coefficient (Ward & Hohmann, Eq. 4.19).
#     Supports x as an array (e.g., for numerical integration) and multiple layers.
#     """
#     x = np.atleast_1d(x).astype(np.complex128)
#     permittivity = np.asarray(permittivity, dtype=np.float64)
#     permeability = np.asarray(permeability, dtype=np.float64)
#     conductivity = np.asarray(conductivity, dtype=np.float64)
#     layer_height = np.asarray(layer_height, dtype=np.float64)

#     mu0 = 4 * np.pi * 1e-7
#     eps0 = 8.854e-12
#     k0_sq = angfreq**2 * mu0 * eps0
#     z0 = 1j * angfreq * mu0

#     # Compute per-layer propagation constants and admittances
#     kn_sq = (angfreq**2 * np.outer(np.ones_like(x), permittivity * permeability)
#              - 1j * angfreq * np.outer(np.ones_like(x), conductivity * permeability))
#     u = np.sqrt(x[:, None]**2 - kn_sq+0j)
#     z = 1j * angfreq * permeability
#     y = u / z

#     # Upward recursion (vectorized over x)
#     y_hat = y[:, -1].copy()
#     for n in range(len(permeability) - 2, -1, -1):
#         e = np.exp(-2 * u[:, n] * layer_height[n])
#         y_hat = y[:, n] * (y_hat * (1 + e) + y[:, n] * (1 - e)) / (y[:, n] * (1 + e) + y_hat * (1 - e))

#     # Reflection coefficient
#     u0 = np.sqrt(x**2 - k0_sq)
#     y0 = u0 / z0
#     return (y0 - y_hat) / (y0 + y_hat)

def rte_function_vectorized(x, angfreq, permittivity, permeability, conductivity, layer_height):
    """
    Fully vectorized Transverse Electric reflection coefficient (Ward & Hohmann, Eq. 4.19).

    Parameters
    ----------
    x : ndarray (Nx, Nfreq)
        Wavenumber domain samples divided by rho.
    angfreq : ndarray (Nx, Nfreq)
        Angular frequencies (broadcasted with x).
    permittivity : array_like (Nlayer,)
        Layer permittivity values.
    permeability : array_like (Nlayer,)
        Layer permeability values.
    conductivity : array_like (Nlayer,)
        Layer conductivity values.
    layer_height : array_like (Nlayer-1,)
        Layer thicknesses (last layer infinite).

    Returns
    -------
    r_transverse_electric : ndarray (Nx, Nfreq)
        Complex TE reflection coefficients for all x, ω.
    """
    # -------------------------------------------------------------------------
    # Constants
    mu0 = 4 * np.pi * 1e-7
    eps0 = 8.854e-12

    # -------------------------------------------------------------------------
    # Free-space parameters (broadcasted)
    k0_sq = angfreq**2 * mu0 * eps0
    u0 = np.sqrt(x**2 - k0_sq + 0j)
    y0 = u0 / (1j * angfreq * mu0)  # admittance of free space

    # -------------------------------------------------------------------------
    # Bottom layer initialization (layer N)
    Nlayer = len(permeability)
    n = Nlayer - 1

    kn_sq = (angfreq**2 * permittivity[n] * permeability[n]
             - 1j * angfreq * conductivity[n] * permeability[n])
    un = np.sqrt(x**2 - kn_sq + 0j)
    yn = un / (1j * angfreq * permeability[n])

    # Initialize upward recursion
    y_hat = yn

    # -------------------------------------------------------------------------
    # Recursive upward reflection combination for all layers
    for n in range(Nlayer - 2, -1, -1):
        kn_sq = (angfreq**2 * permittivity[n] * permeability[n]
                 - 1j * angfreq * conductivity[n] * permeability[n])
        un = np.sqrt(x**2 - kn_sq + 0j)
        yn = un / (1j * angfreq * permeability[n])

        # Exponential term, fully broadcasted
        e = np.exp(-2 * un * layer_height[n])

        # Recursive impedance combination, vectorized
        num = y_hat * (1 + e) + yn * (1 - e)
        den = yn * (1 + e) + y_hat * (1 - e)
        y_hat = yn * (num / den)

    # -------------------------------------------------------------------------
    # Final TE reflection coefficient (vectorized over x, ω)
    r_transverse_electric = (y0 - y_hat) / (y0 + y_hat)
    return r_transverse_electric


def forward_problem_mag_dipole_hz(angfreq, rho, mag_mom, htx, zrx, permeability, 
                                  permittivity, conductivity, layer_height):

    '''
        compute the secondary magnetic field Hz on the air at height zRx from the
        ground surface. The source is a vertical magnetic dipole
    Input: 
        angfreq: angular frequency
        rho: horizontal distance from the Tx to Rx
        magMon: magnetic moment, a given constant
        htx: height of Tx, from the ground
        zrx: height of Rx, from the ground
        permeability, permittivity, conductivity: vectors of magnetic permeability, permittivity and conductivity of layered medium. 
        layerHeight: a vector of the heights of layers, except the deepest layer, which is assumed to be of infinite height.

    Output: 
        I: z-component of the magnetic field at Rx 
        Method: use the integral for permeability for Hz (see the book of Ward & Hoffman,
        eq (4.46)
    =========================================================================
    '''
    scaling_factor = 1
    #angfreq = angfreq
    # Fundamental constants (for soil)
    permeability_0 = 4e-7 * np.pi  # In Henries/met
    permittivity_0 = 8.854e-12  # In Farads/meter

    # Necessary constants for calculating rTE
    k_0 = np.sqrt(angfreq**2 * permeability_0 * permittivity_0)  # Wavenumber in free space
    
    def u_0(x):
        return np.sqrt(x**2 - k_0**2+0j)  # Propagation constant in free space with 0j to ensure complex sqrt

     # Integrand for the modified W-transform method
    def f(x):
        return (
            sp.jv(0, x)
            * ((x / rho) ** 3)
            / (u_0(x / rho))
            / rho
            * rte_function(x / rho, angfreq, permittivity, permeability, conductivity, layer_height)
            * np.exp(u_0(x / rho) * (zrx - htx))
        )
    
    integral = mw_j0_integral(lambda x: f(x) * scaling_factor)
    return (mag_mom/(4*np.pi)*integral / scaling_factor) #Final result



def forward_problem_mag_dipole_hz_vectorized(angfreqs, rho, mag_mom, htx, zrx,
                                             permeability, permittivity, conductivity, layer_height):
    ''' 
        compute the secondary magnetic field Hz on the air at height zRx from the
        ground surface. The source is a vertical magnetic dipole
    Input: 
        angfreq: angular frequency
        rho: horizontal distance from the Tx to Rx
        magMon: magnetic moment, a given constant
        htx: height of Tx, from the ground
        zrx: height of Rx, from the ground
        permeability, permittivity, conductivity: vectors of magnetic permeability, permittivity and conductivity of layered medium. 
        layerHeight: a vector of the heights of layers, except the deepest layer, which is assumed to be of infinite height.

    Output: 
        I: z-component of the magnetic field at Rx 
        Method: use the integral for permeability for Hz (see the book of Ward & Hoffman,
        eq (4.46)
    =========================================================================
    '''
    mu0 = 4e-7 * np.pi
    eps0 = 8.854e-12
    k0 = np.sqrt(angfreqs**2 * mu0 * eps0)

    def integrand(x, idx):
        u0 = np.sqrt((x/rho)**2 - k0[idx]**2 + 0j)
        rte = rte_function_vectorized(x/rho, angfreqs[idx], permittivity, permeability, conductivity, layer_height)
        return sp.jv(0, x) * ((x/rho)**3) * rte / (u0 * rho) * np.exp(u0 * (zrx - htx))

    integral = mw_j0_integral_vectorized(integrand, angfreqs)
    Hz = mag_mom / (4*np.pi) * integral
    return Hz

# def forward_problem_mag_dipole_hz_vectorized(angfreqs, rho, mag_mom, htx, zrx,
#                                              permeability, permittivity, conductivity, layer_height):
#     """
#     Fully vectorized forward model for vertical magnetic dipole (Hz component)
#     using precomputed integrand and vectorized W-transform.
    
#     Parameters
#     ----------
#     angfreqs : array_like, shape (Nfreq,)
#         Angular frequencies
#     rho : float
#         Horizontal distance from Tx to Rx
#     mag_mom : float
#         Magnetic moment of dipole
#     htx : float
#         Height of Tx above ground
#     zrx : float
#         Height of Rx above ground
#     permeability, permittivity, conductivity, layer_height : arrays
#         Layer properties (1D arrays)
    
#     Returns
#     -------
#     Hz : ndarray, shape (Nfreq,)
#         Secondary magnetic field at Rx for each frequency
#     """
#     mu0 = 4 * np.pi * 1e-7
#     eps0 = 8.854e-12

#     angfreqs = np.atleast_1d(angfreqs)
#     Nfreq = len(angfreqs)

#     # ---------------------------------------------------------------------
#     # Wavenumber sampling
#     Nx = 512
#     x_vals = np.linspace(0, 20, Nx)

#     # Broadcasted mesh for x and frequency
#     X, OMEGA = np.meshgrid(x_vals / rho, angfreqs, indexing='ij')  # shape (Nx, Nfreq)

#     # Free-space propagation constant
#     u0 = np.sqrt(X**2 - (OMEGA**2 * mu0 * eps0) + 0j)

#     # ---------------------------------------------------------------------
#     # Compute TE reflection coefficients (vectorized over x)
#     rte = np.zeros_like(X, dtype=np.complex128)
#     for i in range(Nfreq):
#         rte[:, i] = rte_function_vectorized(X[:, i], angfreqs[i], permittivity, permeability,
#                                             conductivity, layer_height)

#     # ---------------------------------------------------------------------
#     # Compute integrand for Hz
#     integrand = sp.jv(0, x_vals)[:, None] * (X**3) * rte / (u0 * rho) * np.exp(u0 * (zrx - htx))

#     # ---------------------------------------------------------------------
#     # Integrate along x using vectorized W-transform
#     integral = mw_j0_integral_vectorized(integrand, angfreqs)

#     # ---------------------------------------------------------------------
#     # Final Hz field
#     Hz = mag_mom / (4 * np.pi) * integral
#     return Hz



def forward_problem_mag_dipole_hrho(angfreq, rho, mag_mom, htx, zrx,
                                    permeability, permittivity, conductivity, layer_height):
    """
    Compute the secondary magnetic field Hρ in air at height zrx from the ground surface.
    The source is a vertical magnetic dipole.

    Parameters
    ----------
    angfreq : np.array or float
        Angular frequency.
    rho : np.array or float
        Horizontal distance from the transmitter to receiver.
    mag_mom : np.array or float
        Magnetic moment of the transmitter loop.
    htx : np.array or float
        Height of the transmitter above ground.
    zrx : np.array or float
        Height of the receiver above ground.
    permeability, permittivity, conductivity : list or np.array
        Vectors of magnetic permeability, permittivity, and conductivity of each soil layer.
    layer_height : list or np.array
        Vector of the heights of layers (except the deepest one, which is assumed infinite).

    Returns
    -------
    np.array
        ρ-component of the secondary magnetic field at the receiver (Hρ).

    Notes
    -----
    Uses the integral formulation for Hρ (Ward & Hohmann, eq. 4.45)
    and evaluates it using the modified W-transform method of Sidi (1988).
    =========================================================================
    """

    scaling_factor = 1
    angfreq = angfreq.to(np.complex128)

    # Fundamental constants (soil)
    permeability_0 = 4e-7 * np.pi  # Henries/m
    permittivity_0 = 8.854e-12        # Farads/m

    # Free-space wavenumber
    k_0 = np.sqrt(angfreq**2 * permeability_0 * permittivity_0)

    def u_0(x):
        return np.sqrt(x**2 - k_0**2)

    # Integrand for the modified W-transform
    def f(x):
        return (
            sp.jv(1, x) *
            ((x / rho) ** 2) / rho *
            rte_function(x / rho, angfreq, permittivity, permeability, conductivity, layer_height) *
            np.exp(u_0(x / rho) * (zrx - htx))
        )

    # Perform integration using modified W-transform for J1
    integral = mw_j1_integral(lambda x: f(x) * scaling_factor)

    # Final result
    return mag_mom / (4 * np.pi) * integral / scaling_factor

def forward_problem_mag_dipole_hz_block_parallel(
    angfreqs, rho, mag_mom, htx, zrx,
    permeability, permittivity, conductivity, layer_height,
    block_size=2, n_jobs=-1
):
    """
    Compute Hz for large frequency arrays using vectorization in blocks
    and parallel execution across blocks.
    """
    # Split angfreqs into blocks
    blocks = [angfreqs[i:i+block_size] for i in range(0, len(angfreqs), block_size)]

    # Helper to compute one block using vectorized function
    def compute_block(freq_block):
        return forward_problem_mag_dipole_hz_vectorized(
            freq_block, rho, mag_mom, htx, zrx,
            permeability, permittivity, conductivity, layer_height
        )

    # Compute all blocks in parallel
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(compute_block)(blk) for blk in blocks
    )

    # Concatenate all results
    return np.concatenate(results)