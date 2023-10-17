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

def velocity_model(state, u, dt):
    """Calculates the movement of the robot (displacement) based on the circular arc velocity model."""
    # Extract linear velocity and yaw rate
    v, omega = u

    # Extract heading
    theta = state[2]

    # Displacement from the velocity model - circular arc model
    dx = (-v/omega) * np.sin(theta) + (v/omega) * np.sin(theta + omega*dt)
    dy = (v/omega) * np.cos(theta) - (v/omega) * np.cos(theta + omega*dt)
    dtheta = omega*dt
    displacement = np.array([dx, dy, dtheta]).reshape((3, 1))

    return displacement

# Step 2
def predictState(state, u, dt):
    """Predicts the new pose μ of the robot, using a velocity model."""

    displacement = velocity_model(state, u, dt)

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
