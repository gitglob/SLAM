# Standard
# External
import numpy as np
# Local
from utils import velocityModel
from .utils import getSigmaPoints, getWeight
from simulation import random_seed
from utils import normalize_angle


np.random.seed(random_seed)

# Step 3
def propagateSigmaPoints(sigma_points, u, dt):
    """
    Passes the sigma points through the non-linear process function.
    
    Parameters
    ----------
    sigma_points : np.ndarray
        Set of sigma points of the current iteration.
    u : np.ndarray
        Control input for the vehicle's motion.
    dt : float
        Time step for the motion model.

    Returns
    -------
    np.ndarray
        Propagated sigma points with the given motion model and control input.
    """
    propagated_sigma_points = []

    for sigma_point in sigma_points:
        dx, dy, dtheta = velocityModel(sigma_point, u, dt)
        [x, y, theta] = [sigma_point[0] + dx, sigma_point[1] + dy, sigma_point[2] + dtheta]
        # theta = normalize_angle(theta)
        propagated_sigma_points.append([x, y, theta])

    return np.array(propagated_sigma_points).reshape((len(sigma_points), 3, 1))

# Step 4
def predictState(propagated_sigma_points, lamda, num_dim):
    """
    Computes the weighted mean of the propagated sigma points to predict the state.
    
    Parameters
    ----------
    propagated_sigma_points : np.ndarray
        Sigma points after being passed through the motion model.
    lamda : float
        Scaling parameter for the sigma points and weights.
    num_dim : int
        Dimension of the state vector.

    Returns
    -------
    np.ndarray
        Predicted state of the system.
    """
    expected_state = np.zeros((num_dim, 1))
    for i in range(2*num_dim + 1):
        expected_state += getWeight(lamda, num_dim, i)[0] * propagated_sigma_points[i]

    return expected_state

# Step 5
def predictStateCov(propagated_sigma_points, expected_state, process_cov, lamda, num_dim):
    """
    Computes the state covariance based on the propagated sigma points.
    
    Parameters
    ----------
    propagated_sigma_points : np.ndarray
        Sigma points after being passed through the motion model.
    expected_state : np.ndarray
        Predicted mean state of the system.
    process_cov : np.ndarray
        Process covariance matrix.
    lamda : float
        Scaling parameter for the sigma points and weights.
    num_dim : int
        Dimension of the state vector.

    Returns
    -------
    np.ndarray
        Predicted state covariance matrix.
    """
    expected_state_cov = np.zeros((num_dim, num_dim))
    for i in range(2*num_dim + 1):
        expected_state_cov += getWeight(lamda, num_dim, i)[1] * (propagated_sigma_points[i] - expected_state) @ (propagated_sigma_points[i] - expected_state).T + process_cov

    return expected_state_cov

# Prediction step
def predict(state, state_cov, u, process_cov, num_dim, lamda, dt):
    """Performs the prediction steps of the EKF SLAM algorithm."""

    # Step 2: Get Sigma Points
    sigma_points = getSigmaPoints(state, lamda, state_cov)

    # Step 3: Predict Sigma Points
    propagated_sigma_points = propagateSigmaPoints(sigma_points, u, dt)

    # Step 4: Predict state
    expected_state = predictState(propagated_sigma_points, lamda, num_dim)

    # Step 5: Predict state covariance
    expected_state_cov = predictStateCov(propagated_sigma_points, expected_state, process_cov, lamda, num_dim)

    return expected_state, expected_state_cov
