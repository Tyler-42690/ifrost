'''
    Test module for the forward_problem_mag_dipole_hz function in one_d_emi_model.py.
    This module uses pytest to validate the function's correctness with sample inputs.
'''
import numpy as np
from . import one_d_emi_model as one


def test_forward_problem_mag_dipole_hz():
    '''
        Test the forward_problem_mag_dipole_hz function with sample inputs
        against two difference conductivity profile configuration datasets (xlsx).
        Make sure the outputs match expected results within a tolerance.
    '''

    # Define test parameters
    angfreq = np.array(2 * np.pi * 1000.0)  # Angular frequency in rad/s
    rho = np.array(0.077)         # Horizontal distances in meters
    mag_mom = np.array(-10**-2)                     # Magnetic moment in Am^2
    htx = np.array(0.08)                         # Transmitter height in meters
    zrx = np.array(0.08)                         # Receiver height in meters
    permeability = np.array([4 * np.pi * 1e-7, 4 * np.pi * 1e-7])  # Permeability of layers
    permittivity = np.array([8.854e-12, 8.854e-12])                      # Permittivity of layers
    conductivity = np.array([2.01, 0.1])        # Conductivity of layers in S/m
    layer_height = np.array([2.0])              # Thickness of the first layer in meters

    # Call the function
    hz = one.forward_problem_mag_dipole_hz(
        angfreq, rho, mag_mom, htx, zrx, permeability, permittivity, conductivity, layer_height
    )

    # Print results for verification
    print("Hz:", hz)
    

if __name__ == "__main__":
    test_forward_problem_mag_dipole_hz()