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
def updateStateCov(C, expected_inf_matrix, measurement_cov):
    """Updates the state uncertainty using the previous state covariance matrix."""
    expected_state_cov = (C.T @ np.linalg.inv(measurement_cov) @ C) + expected_inf_matrix

    return expected_state_cov

# Step 5
def updateState(C, expected_inf_vector, z, measurement_cov):
    """Updates the state using the previous state prediction."""
    expected_state = C.T @ np.linalg.inv(measurement_cov) @ z + expected_inf_vector

    return expected_state

# Correction step
def correct(expected_inf_matrix, expected_inf_vector, z, measurement_cov):
    """Performs the correction steps of the EKF SLAM algorithm."""

    # Get matrix that maps the state to the measurement
    C = getC()

    # Step 4: Uncertainty Update
    inf_matrix = updateStateCov(C, expected_inf_matrix, measurement_cov)

    # Step 5: State update
    inf_vector = updateState(C, expected_inf_vector, z, measurement_cov)

    # Step 6: return
    return inf_matrix, inf_vector
