# Standard
from scipy.linalg import sqrtm
# External
import numpy as np
# Local

def getSigmaPoints(state, state_cov, gamma):
    """
    Creates 2n+1 Sigma Points based on the dimensionality and the scaling parameter lamda.
    
    Parameters
    ----------
    state : np.ndarray
        Current state estimate of the system.
    state_cov : np.ndarray
        Current state covariance matrix.
    gamma : float
        Scaling parameter for the sigma points.

    Returns
    -------
    np.ndarray
        Generated sigma points based on the given parameters.
    """
    # Get the state dimensions
    num_dim = len(state)

    # square root of the covariance matrix
    sqrt_matrix = sqrtm(gamma * state_cov)
    # sqrt_matrix = np.linalg.cholesky(gamma * state_cov)

    # Calculate sigma points
    sigma_points = np.zeros((2*num_dim+1, num_dim, 1))
    sigma_points[0] = state
    for i in range(num_dim):
        sigma_points[i+1] = state + sqrt_matrix[:,i].reshape((num_dim, 1))
        sigma_points[num_dim+i+1] = state - sqrt_matrix[:,i].reshape((num_dim, 1))

    return sigma_points

def getWeight(lamda, gamma, i):
    """
    Calculates the weight for the sigma point with the given index i.
    
    Parameters
    ----------
    lamda : float
        Scaling parameter for the sigma points.
    gamma : int
        Scaling parameter for the sigma points.
    i : int
        Index of the sigma point for which the weight needs to be computed.

    Returns
    -------
    float
        Weight for computing the mean and variance.
    """
    
    if i == 0:
        return lamda / gamma
    else:
        return 1 / (2 * gamma)
