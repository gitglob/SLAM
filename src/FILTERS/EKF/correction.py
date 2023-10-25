# Standard
# External
import numpy as np
# Local
from src.simulation import random_seed
from src.utils import normalize_angle

np.random.seed(random_seed)

# Measurement (non-linear) function
def h(state):
    """
    Nonlinear measurement function mapping state space into measurement space.

    As an example, I assume that I have polar coordinate measurements and I transform them to cartesian.

    Parameters
    ----------
    state : np.ndarray
        The system state vector.
    
    Returns
    -------
    measurement : np.ndarray
        The expected measurement vector.
    """
    x, y, theta = state
    
    range = np.sqrt(x**2 + y**2)
    bearing = np.arctan2(y, x)
    bearing = normalize_angle(bearing)
        
    return np.array([range, bearing]).reshape((2,1))

def getH(state):
    """
    Calculate the Jacobian matrix H of the measurement function h.
    
    Parameters
    ----------
    state : array_like
        The system state vector [x, y, theta].
    
    Returns
    -------
    np.array
        The Jacobian matrix H.
    """
    x, y, _ = state
    
    # Compute the denominators for the partial derivatives
    denom1 = np.sqrt(x**2 + y**2)
    denom2 = x**2 + y**2
    
    # Check for division by zero
    epsilon = 1e-10
    if denom1 < epsilon or denom2 < epsilon:
        raise ValueError("Division by zero in correction Jacobian calculation.")
    
    # Compute the partial derivatives
    h11 = x / denom1
    h12 = y / denom1
    h21 = -y / denom2
    h22 = x / denom2
    
    H = [[h11.item(), h12.item(),  0],
         [h21.item(), h22.item(),  0]]

    return np.array(H).reshape(2,3)

# Step 4
def getKalmanGain(expected_state_cov, measurement_cov, H):
    """Calculates the Kalman Gain."""

    K = expected_state_cov @ H.T @ np.linalg.inv(H @ expected_state_cov @ H.T + measurement_cov)

    return K

# Step 5
def updateState(expected_state, K, z):
    """Updates the state prediction."""

    innovation = z - h(expected_state)
    innovation[1] = normalize_angle(innovation[1])
    expected_state = expected_state + K @ innovation

    return expected_state

# Step 6
def updateStateCov(K, H, expected_state_cov):
    """Updates the state uncertainty."""

    I = np.eye(3,3)
    expected_state_cov = (I - K @ H) @ expected_state_cov

    return expected_state_cov

# Correction step
def correct(expected_state, expected_state_cov, z, measurement_cov):
    """Performs the correction steps of the EKF SLAM algorithm."""

    # Calculate the Jacobian of the measurement model
    H = getH(expected_state)

    # Step 4: Kalman Gain
    K = getKalmanGain(expected_state_cov, measurement_cov, H)

    # Step 5: State update
    state = updateState(expected_state, K, z)

    # Step 6: Uncertainty Update
    state_cov = updateStateCov(K, H, expected_state_cov)

    # Step 7: return
    return state, state_cov
