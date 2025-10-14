'''
    Test module for the forward_problem_mag_dipole_hz function in one_d_emi_model.py.
    This module uses pytest to validate the function's correctness with sample inputs.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from . import one_d_emi_model as one




def test_forward_problem_mag_dipole_hz():
    '''
        Test the forward_problem_mag_dipole_hz function with sample inputs
        against two difference conductivity profile configuration datasets (xlsx).
        Make sure the outputs match expected results within a tolerance.
    '''

    # Define test parameters three layers
    angfreqs = 2*np.pi*np.logspace(np.log10(3e5), np.log10(6e6), num=99, base=10.0)# Angular frequency in rad/s
    rho = np.array(0.059)         # Horizontal distances in meters
    mag_mom = np.array(-10**-2)                     # Magnetic moment in Am^2
    htx = np.array(0.08)                         # Transmitter height in meters
    zrx = -htx                         # Receiver height in meters
    permeability_0 = 4 * np.pi * 10**(-7); # Henries/meter
    permittivity_0 = 8.854 * 10**(-12); # Farads/meter

    permeability = permeability_0*np.array([1, 1, 1])  # Permeability of layers
    permittivity = permittivity_0*np.array([1, 1, 1])                      # Permittivity of layers
    conductivity = 1e-3*np.array([2500, 1008, 16])        # Conductivity of layers in S/m
    layer_height = np.array([0.1, 0.125])              # Thickness of the first layer in meters

    # Compare with using all frequencies at once
    time_start = time.time()
    hz_sim = one.forward_problem_mag_dipole_hz_numba(
        angfreqs, rho, mag_mom, htx, zrx, permeability, permittivity, conductivity, layer_height
    )
    time_end = time.time()
    print(f"JIT computation time: {time_end - time_start:.4f} seconds")
    
    # Load reference data for comparison (3layerver.xlsx) - precomputed results
    df = pd.read_excel('src/ifrost/models/3layerver.xlsx')
    freq = df["Frequency"].to_numpy()
    in_phase = df["In Phase"].to_numpy()
    quadrature = df["Quadrature"].to_numpy()
    
    hz_in_phase = np.real(hz_sim)
    hz_quadrature = np.imag(hz_sim)

    
    

    plt.figure(figsize=(8,6))
    plt.semilogx(freq, in_phase, '-', label='MATLAB Model In-Phase')
    plt.semilogx(freq, hz_in_phase, '--', label='Python Model In-Phase')
    plt.semilogx(freq, quadrature, '-', label='MATLAB Model Quadrature')
    plt.semilogx(freq, hz_quadrature, '--', label='Python Model Quadrature')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnetic Field (A/m)")
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()

def test_forward_problem_mag_dipole_hz_parallel():
    '''
        Test the forward_problem_mag_dipole_hz function with sample inputs
        using parallel computation across frequencies.
    '''

    # Define test parameters for three layers
    angfreqs = 2*np.pi*np.logspace(np.log10(3e5), np.log10(6e6), num=99, base=10.0)  # Angular frequency in rad/s
    rho = 0.059         # Horizontal distance in meters (scalar)
    mag_mom = -10**-2   # Magnetic moment in Am^2 (scalar)
    htx = 0.08           # Transmitter height in meters
    zrx = -htx           # Receiver height in meters

    permeability_0 = 4 * np.pi * 10**(-7)  # Henries/meter
    permittivity_0 = 8.854 * 10**(-12)     # Farads/meter

    permeability = permeability_0 * np.array([1, 1, 1], dtype=np.float32)  # Permeability of layers
    permittivity = permittivity_0 * np.array([1, 1, 1], dtype=np.float32)  # Permittivity of layers
    conductivity = 1e-3 * np.array([2500, 1008, 16], dtype=np.float32)     # Conductivity of layers in S/m
    layer_height = np.array([0.1, 0.125], dtype=np.float32)                # Thickness of first layers

    # -------------------------------------------------------------------------
    # Parallel computation of Hz for each frequency
    time_start = time.time()
    hz_parallel = Parallel(n_jobs=-1,backend="loky")(
        delayed(one.forward_problem_mag_dipole_hz)(
            freq, rho, mag_mom, htx, zrx, permeability, permittivity, conductivity, layer_height
        ) for freq in angfreqs
    )
    time_end = time.time()
    hz_parallel = np.array(hz_parallel)
    print(f"Parallel computation time: {time_end - time_start:.4f} seconds")

    # -------------------------------------------------------------------------
    # Load reference data for comparison (3layerver.xlsx)
    df = pd.read_excel('src/ifrost/models/3layerver.xlsx')
    freq = df["Frequency"].to_numpy()
    in_phase = df["In Phase"].to_numpy()
    quadrature = df["Quadrature"].to_numpy()

    hz_in_phase = np.real(hz_parallel)
    hz_quadrature = np.imag(hz_parallel)

    # -------------------------------------------------------------------------
    # Plot comparison
    plt.figure(figsize=(8,6))
    plt.semilogx(freq, in_phase, '-', label='MATLAB Model In-Phase')
    plt.semilogx(freq, hz_in_phase, '--', label='Python Model In-Phase')
    plt.semilogx(freq, quadrature, '-', label='MATLAB Model Quadrature')
    plt.semilogx(freq, hz_quadrature, '--', label='Python Model Quadrature')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnetic Field (A/m)")
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()

def test_forward_problem_mag_dipole_hz_hybrid():
    '''
        Test the forward_problem_mag_dipole_hz function with sample inputs
        against two difference conductivity profile configuration datasets (xlsx).
        Make sure the outputs match expected results within a tolerance.
    '''

    # Define test parameters three layers
    angfreqs = 2*np.pi*np.logspace(np.log10(3e5), np.log10(6e6), num=99, base=10.0)# Angular frequency in rad/s
    rho = np.array(0.059)         # Horizontal distances in meters
    mag_mom = np.array(-10**-2)                     # Magnetic moment in Am^2
    htx = np.array(0.08)                         # Transmitter height in meters
    zrx = -htx                         # Receiver height in meters
    permeability_0 = 4 * np.pi * 10**(-7); # Henries/meter
    permittivity_0 = 8.854 * 10**(-12); # Farads/meter

    permeability = permeability_0*np.array([1, 1, 1])  # Permeability of layers
    permittivity = permittivity_0*np.array([1, 1, 1])                      # Permittivity of layers
    conductivity = 1e-3*np.array([2500, 1008, 16])        # Conductivity of layers in S/m
    layer_height = np.array([0.1, 0.125])              # Thickness of the first layer in meters

    # Compare with using all frequencies at once
    time_start = time.time()
    hz_sim = one.forward_problem_mag_dipole_hz_block_parallel(
        angfreqs, rho, mag_mom, htx, zrx, permeability, permittivity, conductivity, layer_height
    )
    time_end = time.time()
    print(f"Hybrid computation time: {time_end - time_start:.4f} seconds")
    
    # Load reference data for comparison (3layerver.xlsx) - precomputed results
    df = pd.read_excel('src/ifrost/models/3layerver.xlsx')
    freq = df["Frequency"].to_numpy()
    in_phase = df["In Phase"].to_numpy()
    quadrature = df["Quadrature"].to_numpy()
    
    hz_in_phase = np.real(hz_sim)
    hz_quadrature = np.imag(hz_sim)

    
    

    plt.figure(figsize=(8,6))
    plt.semilogx(freq, in_phase, '-', label='MATLAB Model In-Phase')
    plt.semilogx(freq, hz_in_phase, '--', label='Python Model In-Phase')
    plt.semilogx(freq, quadrature, '-', label='MATLAB Model Quadrature')
    plt.semilogx(freq, hz_quadrature, '--', label='Python Model Quadrature')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnetic Field (A/m)")
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()

if __name__ == "__main__":
    #test_forward_problem_mag_dipole_hz() # Vectorized
    test_forward_problem_mag_dipole_hz_parallel()   # Parallel
    test_forward_problem_mag_dipole_hz_hybrid() # Hybrid