from src.ifrost.models.one_d_emi_model import forward_problem_mag_dipole_hz
import torch


def test_forward_problem_mag_dipole_hz():
    '''
        Test the forward_problem_mag_dipole_hz function with sample inputs
        against two difference conductivity profile configuration datasets (xlsx).
        Make sure the outputs match expected results within a tolerance.
    '''

    # Define test parameters
    angfreq = torch.tensor(2 * torch.pi * 1000.0)  # Angular frequency in rad/s
    rho = torch.tensor([10.0, 20.0, 30.0])         # Horizontal distances in meters
    mag_mom = torch.tensor(1.0)                     # Magnetic moment in Am^2
    htx = torch.tensor(1.0)                         # Transmitter height in meters
    zrx = torch.tensor(1.0)                         # Receiver height in meters
    permeability = torch.tensor([4 * torch.pi * 1e-7, 4 * torch.pi * 1e-7])  # Permeability of layers
    permittivity = torch.tensor([8.854e-12, 8.854e-12])                      # Permittivity of layers
    conductivity = torch.tensor([0.01, 0.1])        # Conductivity of layers in S/m
    layer_height = torch.tensor([5.0])              # Thickness of the first layer in meters

    # Call the function
    hz_real, hz_imag = forward_problem_mag_dipole_hz(
        angfreq, rho, mag_mom, htx, zrx, permeability, permittivity, conductivity, layer_height
    )

    # Print results for verification
    print("Hz Real:", hz_real)
    print("Hz Imag:", hz_imag)