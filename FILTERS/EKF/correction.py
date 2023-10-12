# Standard
# External
import numpy as np
# Local


def getH():
    """Returns the Jacobian of the expected observation model, which is a simple linear model, 
    assuming we can directly measure the position and orientation of the robot."""
    H = np.eye(3)

    return H

# Step 4
def getKalmanGain(expected_state_cov, measurement_cov, H):
    """Calculates the Kalman Gain."""

    K = expected_state_cov @ H.T @ np.linalg.inv(H @ expected_state_cov @ H.T + measurement_cov)

    return K

# Step 5
def updateState(expected_state, K, z, H):
    """Updates the state prediction using the previous state prediction, the Kalman Gain and the real and expected observation of a specific landmark."""

    expected_state = expected_state + K @ (z - H@expected_state)

    return expected_state

# Step 6
def updateStateCov(K, H, expected_state_cov):
    """Updates the state uncertainty using the Kalman Gain and the Jacobian of the function that computes the expected observation."""

    I = np.eye(3,3)
    expected_state_cov = (I - K @ H) @ expected_state_cov

    return expected_state_cov

# Correction step
def correct(expected_state, expected_state_cov, z, measurement_cov):
    """Performs the correction steps of the EKF SLAM algorithm."""

    # Calculate the Jacobian of the measurement model
    H = getH()

    # Step 4: Kalman Fain
    K = getKalmanGain(expected_state_cov, measurement_cov, H)

    # Step 5: State update
    state = updateState(expected_state, K, z, H)

    # Step 6: Uncertainty Update
    state_cov = updateStateCov(K, H, expected_state_cov)

    # Step 7: return
    return state, state_cov
