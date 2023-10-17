# Standard
# External
import numpy as np
# Local

def getG(state, u, dt):
    """Calculates the Jacobian of the motion model at the given state under the current control input."""

    # Extract linear velocity and yaw rate
    v, omega = u

    # Extract heading
    theta = state[2]

    # Jacobian of the motion model
    G = np.eye(3,3)
    G[0,2] = (-v/omega) * np.cos(theta) + (v/omega) * np.cos(theta + omega*dt)
    G[1,2] = (-v/omega) * np.sin(theta) - (v/omega) * np.sin(theta + omega*dt)

    return G

def velocity_model(inf_vector, u, dt):
    """Calculates the movement of the robot (displacement) based on the circular arc velocity model."""
    # Extract linear velocity and yaw rate
    v, omega = u

    # Extract heading
    theta = inf_vector[2]

    # Displacement from the velocity model - circular arc model
    dx = (-v/omega) * np.sin(theta) + (v/omega) * np.sin(theta + omega*dt)
    dy = (v/omega) * np.cos(theta) - (v/omega) * np.cos(theta + omega*dt)
    dtheta = omega*dt
    displacement = np.array([dx, dy, dtheta]).reshape((3, 1))

    return displacement

# Step 2
def infVector2state(inf_matrix, inf_vector):
    """Converts an information vector to a moment/standard state mean using the informatin matrix."""
    state = np.linalg.inv(inf_matrix) @ inf_vector

    return state

# Step 3
def predictCovariance(inf_matrix, process_cov, G):
    """Predicts the new information matrix."""

    expected_inf_matrix = np.linalg.inv(G @ np.linalg.inv(inf_matrix) @ G.T + process_cov)

    return expected_inf_matrix

# Step 4
def predictState(state, u, dt):
    """Predicts the new pose μ of the robot, using a velocity model."""

    displacement = velocity_model(state, u, dt)

    expected_inf_vector = state + displacement

    return expected_inf_vector

# Step 5
def state2infVector(state, inf_matrix):
    """Converts a moment/standard state mean to an information vector using the informatin matrix."""
    inf_vector = inf_matrix @ state

    return inf_vector

# Prediction step
def predict(inf_matrix, inf_vector, state, u, process_cov, dt):
    """Performs the prediction steps of the EKF SLAM algorithm."""

    # Calculate the Jacobian of the motion model
    G = getG(state, u, dt)

    # Step 2: Convert previous inf vector to state mean
    state = infVector2state(inf_matrix, inf_vector)

    # Step 3: Predict state uncertainty
    expected_inf_matrix = predictCovariance(inf_matrix, process_cov, G)

    # Step 4: Predict state
    expected_state = predictState(state, u, dt)

    # Step 5: Convert state mean to inf vector
    expected_inf_vector = state2infVector(expected_state, expected_inf_matrix)

    return expected_inf_matrix, expected_inf_vector, expected_state
