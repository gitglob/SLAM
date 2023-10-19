# Standard
# External
import numpy as np
# Local


# Measurement (non-linear) function
def h(inf_vector):
    """
    Nonlinear measurement function mapping inf_vector space into measurement space.

    As an example, I assume that I have polar coordinate measurements and I transform them to cartesian.

    Parameters
    ----------
    inf_vector : np.ndarray
        The system inf_vector vector.
    
    Returns
    -------
    measurement : np.ndarray
        The expected measurement vector.
    """
    x, y, theta = inf_vector
    
    range = np.sqrt(x**2 + y**2)
    bearing = np.arctan2(y, x)
    
    return [range, bearing]

# Jacobian of the measurement function
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
    if denom1 == 0 or denom2 == 0:
        raise ValueError("Division by zero in Jacobian calculation.")
    
    # Compute the partial derivatives
    h11 = x / denom1
    h12 = y / denom1
    h21 = -y / denom2
    h22 = x / denom2
    
    H =[[h11.item(), h12.item(), 0],
        [h21.item(), h22.item(), 0]]

    return np.array(H).reshape(2,3)

# Step 5
def updateStateCov(expected_inf_matrix, H, measurement_cov):
    """Updates the inf_vector uncertainty."""
    inf_matrix = expected_inf_matrix + H.T @ measurement_cov @ H

    return inf_matrix

# Step 6
def updateState(expected_inf_vector, H, measurement_cov, z, expected_state):
    """Updates the inf_vector prediction."""

    expected_inf_vector = expected_inf_vector + H.T @ np.linalg.inv(measurement_cov) @ (z - h(expected_state) + H @ expected_state)

    return expected_inf_vector

# Correction step
def correct(expected_inf_matrix, expected_inf_vector, expected_state, z, measurement_cov):
    """Performs the correction steps of the EKF SLAM algorithm."""

    # Calculate the Jacobian of the measurement model
    H = getH(expected_state)

    # Step 5: Uncertainty Update
    inf_matrix = updateStateCov(expected_inf_matrix, H, measurement_cov)

    # Step 6: State update
    inf_vector = updateState(expected_inf_vector, H, measurement_cov, z, expected_state)

    # Step 7: return
    return inf_matrix, inf_vector
