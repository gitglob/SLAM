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

    # Extract linear and rotational displacement
    dtheta1, dr, dtheta2 = displacement

    for i in range(len(sigma_points)):
        # New, predicted sigma point
        sigma_points[i,0] += dr * np.cos(normalize_angle(sigma_points[i,2] + dtheta1))
        sigma_points[i,1] += dr * np.sin(normalize_angle(sigma_points[i,2] + dtheta1))
        sigma_points[i,2] += dtheta1 + dtheta2

        sigma_points[i,2] = normalize_angle(sigma_points[i,2])

    return sigma_points

# Step 4
def predictState(propagated_sigma_points, lamda, gamma, num_dim):
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
    for i in range(len(propagated_sigma_points)):
        w_m = getWeight(lamda, gamma, i)
        expected_state += w_m * propagated_sigma_points[i]
        expected_x += + w_m * np.cos(propagated_sigma_points[i,2])
        expected_y += + w_m * np.sin(propagated_sigma_points[i,2])

    expected_state[2] = normalize_angle(np.arctan2(expected_y, expected_x))

    return expected_state

# Step 5
def predictStateCov(propagated_sigma_points, expected_state, process_cov, lamda, gamma, num_dim):
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
    # Set the motion noise for the landmarks to 0
    motion_noise = np.zeros((num_dim, num_dim))
    motion_noise[:3, :3] = process_cov

    # Calculate the expected state covariance for robot + landmarks
    expected_state_cov = np.zeros((num_dim, num_dim))
    for i in range(len(propagated_sigma_points)):
        w_c = getWeight(lamda, gamma, i)
        sigma_points_diff = propagated_sigma_points[i] - expected_state
        sigma_points_diff[2] = normalize_angle(sigma_points_diff[2])
        expected_state_cov += w_c * sigma_points_diff @ sigma_points_diff.T

    # Add the motion noise to the state covariance
    expected_state_cov += motion_noise

    return expected_state_cov

# Step A: Prediction
def predict(state, state_cov, displacement, process_cov, gamma):
    """Performs the prediction steps of the UKF SLAM algorithm."""

    # Step 1: Get Sigma Points
    sigma_points = getSigmaPoints(state, state_cov, gamma)

    # Extract state dimensions
    num_dim = len(state)
    lamda = gamma - num_dim

    # Step 2: Predict Sigma Points
    propagated_sigma_points = propagateSigmaPoints(sigma_points, displacement)

    # Step 3: Predict state
    expected_state = predictState(propagated_sigma_points, lamda, gamma, num_dim)

    # Step 4: Expected state covariance
    expected_state_cov = predictStateCov(propagated_sigma_points, expected_state, process_cov, lamda, gamma, num_dim)

    # Step 5: Return expected state and covariance
    return expected_state, expected_state_cov
