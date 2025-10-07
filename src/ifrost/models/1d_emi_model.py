import torch
import scipy.special as sp
from src.ifrost.models.utils import mw_j0_integral, mw_j1_integral

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
    angfreq = angfreq.to(torch.complex128)
    # Fundamental constants
    permeability_0 = 4 * torch.pi * 1e-7   # Henries/meter
    permittivity_0 = 8.854e-12          # Farads/meter

    # Free-space parameters
    k0_sq = angfreq**2 * permittivity_0 * permeability_0
    u0 = torch.sqrt(x**2 - k0_sq)
    z0 = 1j * angfreq * permeability_0
    y0 = u0 / z0  # Intrinsic admittance of free space

    # Bottom (deepest) layer parameters
    n = layers - 1
    kn_sq = angfreq**2 * permittivity[n] * permeability[n] - 1j * angfreq * conductivity[n] * permeability[n]
    zn = 1j * angfreq * permeability[n]
    un = torch.sqrt(x**2 - kn_sq)
    yn = un / zn

    # Initialize Ŷ(layers) = Yn(layers)
    y_hat = yn

    # Loop upward from bottom layer to surface
    for n in range(layers - 2, -1, -1):
        kn_sq = angfreq**2 * permittivity[n] * permeability[n] - 1j * angfreq * conductivity[n] * permeability[n]
        zn = 1j * angfreq * permeability[n]
        un = torch.sqrt(x**2 - kn_sq)
        yn = un / zn

        e = torch.exp(-2 * un * layer_height[n])
        y_hat = yn * (y_hat * (1 + e) + yn * (1 - e)) / (yn * (1 + e) + y_hat * (1 - e))

    # Reflection coefficient for TE mode
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
    angfreq = angfreq.to(torch.complex128)
    # Fundamental constants (for soil)
    permeability_0 = 4e-7 * torch.pi  # In Henries/met
    permittivity_0 = 8.854e-12  # In Farads/meter

    # Necessary constants for calculating rTE
    k_0 = torch.sqrt(angfreq**2 * permeability_0 * permittivity_0)  # Wavenumber in free space
    
    def u_0(x):
        return torch.sqrt(x**2 - k_0**2)  # Propagation constant in free space

     # Integrand for the modified W-transform method
    def f(x):
        return (
            sp.jv(0, x)
            * ((x / rho) ** 3)
            / (u_0(x / rho))
            / rho
            * rte_function(x / rho, angfreq, permittivity, permeability, conductivity, layer_height)
            * torch.exp(u_0(x / rho) * (zrx - htx))
        )
    
    integral = mw_j0_integral(lambda x: f(x) * scaling_factor)
    return mag_mom/(4*torch.pi)*integral / scaling_factor #Final result

def forward_problem_mag_dipole_hrho(angfreq, rho, mag_mom, htx, zrx,
                                    permeability, permittivity, conductivity, layer_height):
    """
    Compute the secondary magnetic field Hρ in air at height zrx from the ground surface.
    The source is a vertical magnetic dipole.

    Parameters
    ----------
    angfreq : torch.Tensor or float
        Angular frequency.
    rho : torch.Tensor or float
        Horizontal distance from the transmitter to receiver.
    mag_mom : torch.Tensor or float
        Magnetic moment of the transmitter loop.
    htx : torch.Tensor or float
        Height of the transmitter above ground.
    zrx : torch.Tensor or float
        Height of the receiver above ground.
    permeability, permittivity, conductivity : list or torch.Tensor
        Vectors of magnetic permeability, permittivity, and conductivity of each soil layer.
    layer_height : list or torch.Tensor
        Vector of the heights of layers (except the deepest one, which is assumed infinite).

    Returns
    -------
    torch.Tensor
        ρ-component of the secondary magnetic field at the receiver (Hρ).

    Notes
    -----
    Uses the integral formulation for Hρ (Ward & Hohmann, eq. 4.45)
    and evaluates it using the modified W-transform method of Sidi (1988).
    =========================================================================
    """

    scaling_factor = 1
    angfreq = angfreq.to(torch.complex128)

    # Fundamental constants (soil)
    permeability_0 = 4e-7 * torch.pi  # Henries/m
    permittivity_0 = 8.854e-12        # Farads/m

    # Free-space wavenumber
    k_0 = torch.sqrt(angfreq**2 * permeability_0 * permittivity_0)

    def u_0(x):
        return torch.sqrt(x**2 - k_0**2)

    # Integrand for the modified W-transform
    def f(x):
        return (
            sp.jv(1, x) *
            ((x / rho) ** 2) / rho *
            rte_function(x / rho, angfreq, permittivity, permeability, conductivity, layer_height) *
            torch.exp(u_0(x / rho) * (zrx - htx))
        )

    # Perform integration using modified W-transform for J1
    integral = mw_j1_integral(lambda x: f(x) * scaling_factor)

    # Final result
    return mag_mom / (4 * torch.pi) * integral / scaling_factor
