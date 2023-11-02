# Standard
# External
import numpy as np
# Local
from src.utils import normalize_angle
from .utils import getSigmaPoints, getWeight


# Step 3
def propagateSigmaPoints(sigma_points, displacement):
    """
    Passes the sigma points through the non-linear process function.
    
    Parameters
    ----------
    sigma_points : np.ndarray
        Set of sigma points of the current iteration.
    displacement : np.ndarray
        The displacement from the robot's motion.

    Returns
    -------
    np.ndarray
        Propagated sigma points with the given motion model.
    """
    propagated_sigma_points = np.zeros((len(sigma_points), 3, 1))

    # Extract linear and rotational displacement
    dtheta1, dr, dtheta2 = displacement

    for i, sigma_point in enumerate(sigma_points):
        # New, predicted sigma point
        propagated_sigma_points[i,0] = sigma_point[0] + dr * np.cos(sigma_point[2] + dtheta1)
        propagated_sigma_points[i,1] = sigma_point[1] + dr * np.sin(sigma_point[2] + dtheta1)
        propagated_sigma_points[i,2] = sigma_point[2] + dtheta1 + dtheta2

        # Normalize angle
        propagated_sigma_points[i,2] = normalize_angle(propagated_sigma_points[i,2])

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
    expected_x = 0
    expected_y = 0
    for i in range(2*num_dim + 1):
        w_m = getWeight(lamda, num_dim, i)[0]
        expected_state += w_m * propagated_sigma_points[i]
        expected_x += + w_m * np.cos(propagated_sigma_points[i,2])
        expected_y += + w_m * np.sin(propagated_sigma_points[i,2])

    expected_state[2] = normalize_angle(np.arctan2(expected_y, expected_x))

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
        w_c = getWeight(lamda, num_dim, i)[1]
        expected_state_cov += w_c * (propagated_sigma_points[i] - expected_state) @ (propagated_sigma_points[i] - expected_state).T + process_cov

    return expected_state_cov

# Step A: Prediction
def predict(state, state_cov, displacement, process_cov, num_dim, lamda):
    """Performs the prediction steps of the UKF SLAM algorithm."""

    # Step 1: Get Sigma Points
    sigma_points = getSigmaPoints(state, state_cov, lamda)

    # Step 2: Predict Sigma Points
    propagated_sigma_points = propagateSigmaPoints(sigma_points, displacement)

    # Step 3: Predict state
    expected_state = predictState(propagated_sigma_points, lamda, num_dim)

    # Step 4: Expected state covariance
    expected_state_cov = predictStateCov(propagated_sigma_points, expected_state, process_cov, lamda, num_dim)

    # Step 5: Return expected state and covariance
    return expected_state, expected_state_cov
