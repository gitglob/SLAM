# Standard
# External
import numpy as np
# Local


def getC():
    """Constructs the C matrix, that maps the state to the observations."""

    C = [[1, 0, 0, 0],
         [0, 1, 0, 0]]
    C = np.array(C).reshape((2,4))

    return C

# Step 4
def getKalmanGain(expected_state_cov, measurement_cov):
    """Calculates the Kalman Gain."""

    K = expected_state_cov @ getC().T @ np.linalg.inv(getC() @ expected_state_cov @ getC().T + measurement_cov)

    return K

# Step 5
def updateState(expected_state, K, z):
    """Updates the state prediction using the previous state prediction, the Kalman Gain and the real and expected observation of a specific landmark."""

    expected_state = expected_state + K @ (z - getC()@expected_state)

    return expected_state

# Step 6
def updateStateCov(K, expected_state_cov):
    """Updates the state uncertainty using the Kalman Gain and the Jacobian of the function that computes the expected observation."""

    I = np.eye(4,4)
    expected_state_cov = (I - K @ getC()) @ expected_state_cov

    return expected_state_cov

# Correction step
def correct(expected_state, expected_state_cov, z, measurement_cov):
    """Performs the correction steps of the EKF SLAM algorithm."""

    # Step 4: Kalman Fain
    K = getKalmanGain(expected_state_cov, measurement_cov)

    # Step 5: State update
    state = updateState(expected_state, K, z)

    # Step 6: Uncertainty Update
    state_cov = updateStateCov(K, expected_state_cov)

    # Step 7: return
    return state, state_cov
