# Standard
# External
import numpy as np
# Local
from src.utils import normalize_angle


# Step 3
def predictState(state, Fx, displacement):
    """Predicts the new pose μ of the robot, based on the circular arc velocity model."""

    # Extract linear and rotational displacement
    dtheta1, dr, dtheta2 = displacement

    # Extract heading
    theta = state[2]

    # Displacement from the velocity model - circular arc model
    dx = dr * np.cos(theta + dtheta1)
    dy = dr * np.sin(theta + dtheta1)
    dtheta = dtheta1 + dtheta2
    velocityModel = np.array([dx, dy, dtheta]).reshape((3, 1))

    # Calculate expected state
    expected_state = state + Fx.T @ velocityModel

    # Normalize new bearing
    expected_state[3] = normalize_angle(expected_state[3])

    return expected_state

# Step 4
def getG(state, Fx, displacement, NUM_LANDMARKS):
    """Calculates the Jacobian of the motion model at the given state under the current control input."""

    # Extract linear and rotational displacement
    dtheta1, dr, _ = displacement

    # Extract heading
    theta = state[2]

    # Jacobian of the motion model
    I = np.eye(3 + 2*NUM_LANDMARKS, 3 + 2*NUM_LANDMARKS) # Identity for the 2Nx2N landmark part
    Gx = np.zeros((3,3))
    Gx[0,2] = -dr * np.sin(theta + dtheta1)
    Gx[1,2] = dr * np.cos(theta + dtheta1)

    G = I + Fx.T @ Gx @ Fx

    return G

# Step 5
def predictCovariance(G, state_covariance, Fx, process_cov):
    """Predicts the new covariance matrix."""

    # Extract the process noise that corresponds to the robot's pose
    Rx = process_cov

    # Process noise - only affects the robot, not the landmarks
    R = Fx.T @ Rx @ Fx

    # In the new uncertainty, only the parts that contain the robot's pose will change
    updated_covariance = G @ state_covariance @ G.T

    expected_covariance = updated_covariance + R

    return expected_covariance

# Step A: Prediction
def predict(state, state_covariance, displacement, process_cov, Fx, NUM_LANDMARKS):
    """Performs the prediction steps of the EKF SLAM algorithm."""
    # Step 1: Construct Fx matrix (done during initialization)

    # Step 2: Predict state
    expected_state = predictState(state, Fx, displacement)

    # Step 3: Calculate motion jacobian
    G = getG(state, Fx, displacement, NUM_LANDMARKS)

    # Step 4: Expected state covariance
    expected_state_cov = predictCovariance(G, state_covariance, Fx, process_cov)

    # Step 5: Return expected state and covariance
    return expected_state, expected_state_cov
