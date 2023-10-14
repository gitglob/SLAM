# Standard
from scipy.linalg import sqrtm
# External
import numpy as np
# Local
from . import alpha, beta

def getSigmaPoints(state, num_dim, lamda, state_cov):
    """Creates 2n+1 Sigma Points based on the dimensionality num_dim and the scaling parameter lamda."""
    # Combine the UKF sigma point scale parameters
    gamma = num_dim-lamda

    # square root of the covariance matrix
    state_cov_sq = sqrtm(state_cov)

    # Calculate sigma points
    sigma_points = np.zeros((2*num_dim+1, 3, 1))
    sigma_points[0] = state
    for i in range(num_dim):
        sigma_points[i+1]   = state + gamma * state_cov_sq[:,i].reshape((3,1))
        sigma_points[num_dim+i+1] = state - gamma * state_cov_sq[:,i].reshape((3,1))

    return sigma_points

def getWeight(lamda, num_dim, i, alpha=alpha, beta=beta):
    """Calculates the weight for the sigma point with the given index i.
    
    Parameters
    ----------
    alpha : float
    beta : float 
        scaling parameters
    
    Returns
    -------
    w_m : float
        For computing the mean
    w_c : float
        For computing the variance
    """

    w_m_0 = lamda / (num_dim + lamda)
    w_c_0 = w_m_0 + (1 - alpha**2 + beta)
    
    if i==0:
        return w_m_0, w_c_0
    else:
        w_m = 1 / (2 * (num_dim+lamda))
        w_c = w_m

        return w_m, w_c
    
def getLamda(alpha, kappa, num_dim):
    """Calculates the lamda UKF parameter based on the state mensionality, 
    and the alpha and kappa parameters."""
    lamda = alpha**2 * (num_dim + kappa) - num_dim

    return lamda