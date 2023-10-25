# Standard
# External
import numpy as np
# Local
from src.utils import normalize_angle, xy2polar
from src.simulation import random_seed
from .utils import getSigmaPoints, getWeight

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
    x, y, phi = state
    
    range = np.sqrt(x**2 + y**2).item()
    bearing = np.arctan2(y, x).item()
    
    return np.array([range, bearing]).reshape((2,1))

# Step 7
def propagateSigmaPoints(sigma_points):
    """
    Passes the sigma points through the non-linear measurement function.
    
    Parameters
    ----------
    sigma_points : np.ndarray

    Returns
    -------
    np.ndarray
        The sigma points passed through the non-linear measurement function.
    """
    expected_sigma_points = np.zeros((len(sigma_points), 2, 1))
    for i, sigma_point in enumerate(sigma_points):
        expected_sigma_points[i] = h(sigma_point)

    return expected_sigma_points

# Step 8
def predictObservation(propagated_sigma_points, lamda, num_dim):
    """Calculates the weighted mean of the propagated sigma points."""

    expected_observation = np.zeros((2,1))
    for i in range(2*num_dim+1):
        expected_observation += getWeight(lamda, num_dim, i)[0] * propagated_sigma_points[i]

    return expected_observation

# Step 9
def updateUncertainty(propagated_sigma_points, expected_observation, measurement_cov, lamda, num_dim):
    """Calculates S."""
    S = np.zeros(measurement_cov.shape)
    for i in range(2*num_dim+1):
        S += getWeight(lamda, num_dim, i)[1] * (propagated_sigma_points[i] - expected_observation) @ (propagated_sigma_points[i] - expected_observation).T + measurement_cov

    return S

# Step 10
def predictStateCov(sigma_points, expected_state, propagated_sigma_points, expected_observation, lamda, num_dim):
    """Calculates the expected state uncertainty."""
    expected_state_cov = np.zeros((num_dim, 2))
    for i in range(2*num_dim+1):
        expected_state_cov += getWeight(lamda, num_dim, i)[1] * (sigma_points[i] - expected_state) @ (propagated_sigma_points[i] - expected_observation).T

    return expected_state_cov

# Step 11
def getKalmanGain(expected_state_cov, S):
    """Calculates the Kalman Gain."""
    K = expected_state_cov @ np.linalg.inv(S)

    return K

# Step 12
def updateState(expected_state, K, observation, expected_observation):
    """Updates the robot's state."""
    state = expected_state + K @ (observation - expected_observation)

    return state

# step 13
def updateStateCov(expected_state_cov, K, S):
    """Updates the state covariance matrrix."""
    state_cov = expected_state_cov - K @ S @ K.T

    return state_cov

# Correction step
def correct(expected_state, expected_state_cov, observation, measurement_cov, num_dim, lamda):
    """Performs the correction steps of the EKF SLAM algorithm."""

    # Step 6: Get sigma points
    sigma_points = getSigmaPoints(expected_state, lamda, expected_state_cov)

    # Step 7: State update
    propagated_sigma_points = propagateSigmaPoints(sigma_points)

    # Step 8: Uncertainty Update
    expected_observation = predictObservation(propagated_sigma_points, lamda, num_dim)

    # Step 9: State Update
    S = updateUncertainty(propagated_sigma_points, expected_observation, measurement_cov, lamda, num_dim)

    # Step 10: Uncertainty Update
    expected_cross_cov = predictStateCov(sigma_points, expected_state, propagated_sigma_points, expected_observation, lamda, num_dim)

    # Step 11: Kalman Gain
    K = getKalmanGain(expected_cross_cov, S)

    # Step 12: State
    state = updateState(expected_state, K, observation, expected_observation)

    # step 13: State Uncertainty
    state_cov = updateStateCov(expected_state_cov, K, S)

    # Step 14: return
    return state, state_cov
