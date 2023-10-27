# Standard
from copy import deepcopy
# External
import numpy as np
from numpy.linalg import inv
# Local
from . import NUM_LANDMARKS

# Step 1
def getProjectionMatrices(m_0_idx):
    """
    Constructs the F matrices, which project different states to their observations.
    
    Parameters:
    - m_0_idx (int): Index of the new active landmarks.
    
    Returns:
    - tuple: The F matrices for the m0 state, the combined x and m0 states, and the x state.
    """
    # Initialize the F matrix with zeros
    F = np.zeros((3, 3 + 2*NUM_LANDMARKS))

    # Construct Fx for only the state
    Fx = deepcopy(F)
    Fx[:3,:3] = np.eye(3) 

    # Construct Fm0 for the new active landmarks
    Fm0 = deepcopy(F)
    for idx in m_0_idx:
        Fm0[:3, 3 + 2*idx : 3 + 2*idx + 2] = np.eye(2) 

    # Construct Fxm0 for both state and new active landmarks
    Fxm0 = deepcopy(F)
    Fxm0[:3,:3] = np.eye(3)
    for idx in m_0_idx:
        Fxm0[:3, 3 + 2*idx : 3 + 2*idx + 2] = np.eye(2)

    return Fm0, Fxm0, Fx

# Step 2
def getInfMatrix0(m_active_idx, m_0_idx, inf_matrix):
    """
    Compute a specific representation of the information matrix considering active landmarks and a new active landmark.

    Parameters:
    - m_active_idx (list): Indices of active landmarks.
    - m_0_idx (int): Index of new active landmarks.
    - inf_matrix (numpy.ndarray): The current information matrix.

    Returns:
    - numpy.ndarray: The processed information matrix.
    """
    F = np.zeros((3, 3 + 2*NUM_LANDMARKS))
    
    # Construct the F matrix considering active landmarks and a new active landmarks
    F_x_m_active_m_0 = deepcopy(F)
    F_x_m_active_m_0[:3,:3] = np.eye(3)
    for idx in m_active_idx:
        F_x_m_active_m_0[:3, 3 + 2*idx : 3 + 2*idx + 2] = np.eye(2) 
    for idx in m_0_idx:
        F_x_m_active_m_0[:3, 3 + 2*idx : 3 + 2*idx + 2] = np.eye(2)

    # Compute the transformed information matrix
    inf_matrix0 = F_x_m_active_m_0 @ F_x_m_active_m_0.T @ inf_matrix @ F_x_m_active_m_0 @ F_x_m_active_m_0.T

    return inf_matrix0


# Step 3
def getSparseInfMatrix(inf_matrix, inf_matrix0, Fm0, Fxm0, Fx):
    """
    Compute the sparse representation of the information matrix based on various projection matrices.

    Parameters:
    - inf_matrix (numpy.ndarray): The current information matrix.
    - inf_matrix0 (numpy.ndarray): The information matrix conditioned on passive landmarks.
    - Fm0 (numpy.ndarray): The projection matrix new active landmarks.
    - Fxm0 (numpy.ndarray): The projection matrix for both state and new active landmarks.
    - Fx (numpy.ndarray): The projection matrix for only the state.

    Returns:
    - numpy.ndarray: The sparse representation of the information matrix.
    """
    # Compute various transformations of the information matrix
    inf_matrix1 = inf_matrix - inf_matrix0 @ Fm0 @ inv(Fm0.T @ inf_matrix0 @ Fm0) @ Fm0.T @ inf_matrix0
    inf_matrix2 = inf_matrix0 @ Fxm0 @ inv(Fxm0.T @ inf_matrix0 @ Fxm0) @ Fxm0.T @ inf_matrix0
    inf_matrix3 = inf_matrix @ Fx @ inv(Fx.T @ inf_matrix @ Fx) @ Fx.T @ inf_matrix

    # Combine transformations to achieve the final sparse information matrix
    sparse_inf_matrix = inf_matrix1 - inf_matrix2 + inf_matrix3

    return sparse_inf_matrix

# Step 4
def getSparseInfVector(inf_vector, sparse_inf_matrix, inf_matrix, state):
    """
    Compute the sparse representation of the information vector.

    Parameters:
    - inf_vector (numpy.ndarray): The current information vector.
    - sparse_inf_matrix (numpy.ndarray): The sparse information matrix.
    - inf_matrix (numpy.ndarray): The current information matrix.
    - state (numpy.ndarray): The current state vector.

    Returns:
    - numpy.ndarray: The sparse representation of the information vector.
    """
    # Update the information vector using the sparse and current information matrices and state
    sparse_inf_vector = inf_vector + (sparse_inf_matrix - inf_matrix) @ state

    return sparse_inf_vector

def sparsify(inf_vector, inf_matrix, state):
    # Step 1
    Fm0, Fxm0, Fx = getProjectionMatrices()
    
    # Step 2
    inf_matrix0 = getInfMatrix0(Fm0, Fx, Fx, inf_matrix)
    
    # Step 3
    sparse_inf_matrix = getSparseInfMatrix(inf_matrix, inf_matrix0, Fm0, Fxm0, Fx)
    
    # Step 4
    sparse_inf_vector = getSparseInfVector(inf_vector, sparse_inf_matrix, inf_matrix, state)

    # Step 5
    return sparse_inf_vector, sparse_inf_matrix


def simple_sparsify(inf_matrix, inf_vector, state, threshold=0.01):
    """
    Sparsify the information matrix and vector based on a given threshold.

    This function sets elements of the information matrix and vector to zero
    if their absolute values are below a certain threshold. This results in 
    a sparse representation which can lead to computational benefits.

    Parameters:
    - inf_matrix (numpy.ndarray): The dense information matrix of shape (n, n).
    - inf_vector (numpy.ndarray): The dense information vector of shape (n,1).
    - state (numpy.ndarray): The state of shape (n,1).
    - threshold (float, optional): The threshold below which values in the information 
      matrix and vector will be set to zero. Default is 0.01.

    Returns:
    - tuple: A tuple containing the sparse information matrix and sparse information vector.
    
    Example:
    --------
    inf_matrix = np.array([[4, 0.001], [0.001, 4]])
    inf_vector = np.array([2, 0.005])
    state = np.array([0.5, 0.5])
    sparse_inf_matrix, sparse_inf_vector = sparsify(inf_matrix, inf_vector, state)
    print(sparse_inf_matrix)
    print(sparse_inf_vector)
    """

    # Sparsify the information matrix by setting values below the threshold to zero
    inf_matrix[np.abs(inf_matrix) < threshold] = 0
    
    # Similarly, sparsify the information vector
    inf_vector[np.abs(inf_vector) < threshold] = 0
    
    return inf_matrix, inf_vector