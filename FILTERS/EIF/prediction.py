# Standard
# External
import numpy as np
# Local
from utils import velocityModel
from .utils import canonical2moment

def getG(state, u, dt):
    """Calculates the Jacobian of the motion model at the given state under the current control input."""

    # Extract linear velocity and yaw rate
    v, omega = u

    # Extract heading
    theta = state[2]

    # Jacobian of the motion model
    G = np.eye(3,3)
    G[0,2] = (-v/omega) * np.cos(theta) + (v/omega) * np.cos(theta + omega*dt)
    G[1,2] = (-v/omega) * np.sin(theta) + (v/omega) * np.sin(theta + omega*dt)

    return G

# Step 3
def predictCovariance(inf_matrix, process_cov, G):
    """Predicts the new information matrix."""

    expected_inf_matrix = np.linalg.inv(G @ np.linalg.inv(inf_matrix) @ G.T + process_cov)

    return expected_inf_matrix

# Step 4
def predictState(state, u, dt):
    """Predicts the new pose μ of the robot, using a velocity model."""

    # Displacement based on the velocity model
    displacement = np.vstack(velocityModel(state, u, dt))

    # New, predicted state
    expected_state = state + displacement

    return expected_state

# Step 5
def state2infVector(state, inf_matrix):
    """Converts a moment/standard state mean to an information vector using the informatin matrix."""
    inf_vector = inf_matrix @ state

    return inf_vector

# Prediction step
def predict(inf_matrix, inf_vector, u, process_cov, dt):
    """Performs the prediction steps of the EKF SLAM algorithm."""

    # Step 2: Convert previous inf vector to state mean
    _, state = canonical2moment(inf_matrix, inf_vector)

    # Calculate the Jacobian of the motion model
    G = getG(state, u, dt)

    # Step 3: Predict state uncertainty
    expected_inf_matrix = predictCovariance(inf_matrix, process_cov, G)

    # Step 4: Predict state
    expected_state = predictState(state, u, dt)

    # Step 5: Convert state mean to inf vector
    expected_inf_vector = state2infVector(expected_state, expected_inf_matrix)

    return expected_inf_matrix, expected_inf_vector, expected_state
