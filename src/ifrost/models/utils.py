import torch

torch.set_default_dtype(torch.float64)  # Set precision to 64-bit
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

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