# Standard
# External
import numpy as np
# Local
from .utils import getSigmaPoints, getWeight


def velocity_model(state, u, dt):
    """Calculates the movement of the robot (displacement) based on the circular arc velocity model."""
    # Extract linear velocity and yaw rate
    v, omega = u

    # Extract heading
    theta = state[2]

    # Displacement from the velocity model - circular arc model
    dx = (-v/omega) * np.sin(theta) + (v/omega) * np.sin(theta + omega*dt)
    dy = (v/omega) * np.cos(theta) - (v/omega) * np.cos(theta + omega*dt)
    dz = omega*dt
    displacement = np.array([dx, dy, dz]).reshape((3, 1))

    return displacement

# Step 2
def predictState(state, u, dt):
    """Predicts the new pose μ of the robot, using a velocity model."""

    displacement = velocity_model(state, u, dt)

    expected_state = state + displacement

    return expected_state

# Step 3
def propagateSigmaPoints(sigma_points, u, dt):
    """Passes the sigma points of the current iteration through the non-linear process function."""
    propagated_sigma_points = []
    for sigma_point in sigma_points:
        propagated_sigma_points.append(velocity_model(sigma_point, u, dt).tolist())

    return np.array(propagated_sigma_points).reshape((len(sigma_points), 3, 1))

# Step 4
def predictState(propagated_sigma_points, lamda, num_dim):
    """Calculates the weighted mean of the propagated sigma points."""

    expected_state = np.zeros((num_dim, 1))
    for i in range(0, 2*num_dim):
        expected_state += getWeight(lamda, num_dim, i)[0] * propagated_sigma_points[i]

    return expected_state

# Step 5
def predictStateCov(propagated_sigma_points, expected_state, process_cov, lamda, num_dim):
    """Calculates the expected state uncertainty."""
    expected_state_cov = np.zeros((num_dim, num_dim))
    for i in range(0, 2*num_dim):
        expected_state_cov += getWeight(lamda, num_dim, i)[1] * (propagated_sigma_points[i] - expected_state) @ (propagated_sigma_points[i] - expected_state).T + process_cov

    return expected_state_cov

# Prediction step
def predict(state, state_cov, u, process_cov, num_dim, lamda, dt):
    """Performs the prediction steps of the EKF SLAM algorithm."""

    # Step 2: Get Sigma Points
    sigma_points = getSigmaPoints(state, num_dim, lamda, state_cov)

    # Step 3: Predict Sigma Points
    propagated_sigma_points = propagateSigmaPoints(sigma_points, u, dt)

    # Step 4: Predict state
    expected_state = predictState(propagated_sigma_points, lamda, num_dim)

    # Step 5: Predict state covariance
    expected_state_cov = predictStateCov(propagated_sigma_points, expected_state, process_cov, lamda, num_dim)

    return expected_state, expected_state_cov
