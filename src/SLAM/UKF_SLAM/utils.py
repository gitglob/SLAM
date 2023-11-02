# Standard
from scipy.linalg import sqrtm
# External
import numpy as np
# Local
from . import alpha, beta

def getSigmaPoints(state, state_cov, gamma):
    """
    Creates 2n+1 Sigma Points based on the dimensionality and the scaling parameter lamda.
    
    Parameters
    ----------
    state : np.ndarray
        Current state estimate of the system.
    lamda : float
        Scaling parameter for the sigma points.
    state_cov : np.ndarray
        Current state covariance matrix.

    Returns
    -------
    np.ndarray
        Generated sigma points based on the given parameters.
    """
    # Get the state dimensions
    num_dim = state.shape[0]

    # Combine the UKF sigma point scale parameters
    lamda = gamma - num_dim

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

def getWeight(lamda, num_dim, i, alpha=alpha, beta=beta):
    """
    Calculates the weight for the sigma point with the given index i.
    
    Parameters
    ----------
    lamda : float
        Scaling parameter for the sigma points.
    num_dim : int
        Dimension of the state vector.
    i : int
        Index of the sigma point for which the weight needs to be computed.
    alpha : float
        Scaling parameter for the UKF.
    beta : float
        Scaling parameter representing the knowledge about the Gaussian distribution.

    Returns
    -------
    float, float
        Weights for computing the mean and variance, respectively.
    """
    w_m_0 = lamda / (num_dim + lamda)
    w_c_0 = w_m_0 + (1 - alpha**2 + beta)
    
    if i == 0:
        return w_m_0, w_c_0
    else:
        w_m = 1 / (2 * (num_dim + lamda))
        w_c = w_m

        return w_m, w_c

def getLamda(alpha, kappa, num_dim):
    """
    Calculates the lambda parameter for the UKF based on state dimensionality, 
    and the alpha and kappa scaling parameters.
    
    Parameters
    ----------
    alpha : float
        Scaling parameter for the UKF.
    kappa : float
        Secondary scaling parameter for the UKF.
    num_dim : int
        Dimension of the state vector.

    Returns
    -------
    float
        Calculated lambda parameter.
    """
    lamda = alpha**2 * (num_dim + kappa) - num_dim

    return lamda