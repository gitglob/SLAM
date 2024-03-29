# Standard
# External
import numpy as np
# Local
from src.utils import velocityModel
from src.simulation import random_seed


np.random.seed(random_seed)

def getG(state, u, dt):
    """Calculates the Jacobian of the motion model at the given state under the current control input."""

    # Extract linear velocity and yaw rate
    v, omega = u

    # Extract heading
    theta = state[2]

    # Initialize the Jacobian with identity matrix
    G = np.eye(3, 3)

    # Handle the case when omega is zero (or close to zero) - straight line motion
    epsilon = 1e-6
    if omega < epsilon:
        G[0, 2] = -v * np.sin(theta) * dt
        G[1, 2] = v * np.cos(theta) * dt
    else:
        # Jacobian elements for the general motion model (circular arc model)
        G[0, 2] = (-v/omega) * np.cos(theta) + (v/omega) * np.cos(theta + omega*dt)
        G[1, 2] = (-v/omega) * np.sin(theta) + (v/omega) * np.sin(theta + omega*dt)

    return G

# Step 2
def predictState(state, u, dt):
    """Predicts the new pose μ of the robot, using a velocity model."""

    # Displacement based on the velocity model
    displacement = np.vstack(velocityModel(state, u, dt))

    # New, predicted state
    expected_state = state + displacement

    return expected_state

# Step 3
def predictCovariance(state_covariance, process_cov, G):
    """Predicts the new covariance matrix."""

    expected_covariance = G@state_covariance@G.T + process_cov

    return expected_covariance

# Prediction step
def predict(state, state_covariance, u, process_cov, dt):
    """Performs the prediction steps of the EKF SLAM algorithm."""

    # Calculate the Jacobian of the motion model
    G = getG(state, u, dt)

    # Step 2: Predict state
    expected_state = predictState(state, u, dt)

    # Step 3: Expected state covariance
    expected_state_cov = predictCovariance(state_covariance, process_cov, G)

    return expected_state, expected_state_cov
