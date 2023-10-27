# External
import numpy as np
# Local
from . import NUM_LANDMARKS


# Step 2
def getFx():
    """Calculates the Fx matrix, which is essential to apply the motion model to the pose state, and not the landmarks.
    Fx maps the robot's state from a low dimensional space, to a high dimensional space."""

    Fx = np.zeros((3, 3 + 2*NUM_LANDMARKS))
    Fx[:3, :3] = np.eye(3)

    return Fx

# Step 3
def predictState(state, Fx, u, dt):
    """Predicts the new pose μ of the robot, based on the circular arc velocity model."""

    # Extract linear velocity and yaw rate
    v, omega = u

    # Extract heading
    theta = state[2]

    # Displacement from the velocity model - circular arc model
    dx = (-v/omega) * np.sin(theta) + (v/omega) * np.sin(theta + omega*dt)
    dy = (v/omega) * np.cos(theta) - (v/omega) * np.cos(theta + omega*dt)
    dtheta = omega*dt
    velocityModel = np.array([dx, dy, dtheta]).reshape((3, 1))

    expected_state = state + Fx.T @ velocityModel

    return expected_state

# Step 4
def getG(state, Fx, u, dt):
    """Calculates the Jacobian of the motion model at the given state under the current control input."""

    # Extract linear velocity and yaw rate
    v, omega = u

    # Extract heading
    theta = state[2]

    # Jacobian of the motion model
    I = np.eye(3 + 2*NUM_LANDMARKS, 3 + 2*NUM_LANDMARKS) # Identity for the 2Nx2N landmark part
    G_t_x_deriv = (-v/omega) * np.cos(theta) + (v/omega) * np.cos(theta + omega*dt)
    G_t_y_deriv = (-v/omega) * np.sin(theta) - (v/omega) * np.sin(theta + omega*dt)
    velocity_model_deriv = np.vstack((G_t_x_deriv, G_t_y_deriv, 0))
    G_t_x = np.hstack((np.zeros((3,1)), np.zeros((3,1)), velocity_model_deriv)) # Jacobian of the motion

    G_t = I + Fx.T @ G_t_x @ Fx

    return G_t

# Step 5
def predictCovariance(G_t, state_covariance, Fx, process_cov):
    """Predicts the new covariance matrix."""

    # Extract the process noise that corresponds to the robot's pose
    R_t_x = process_cov[:3, :3]

    # Process noise - only affects the robot, not the landmarks
    R_t = Fx.T @ R_t_x @ Fx

    # In the new uncertainty, only the parts that contain the robot's pose will change
    updated_covariance = G_t @ state_covariance @ G_t.T

    expected_covariance = updated_covariance + R_t

    return expected_covariance

# Step 1: Prediction
def predict(state, state_covariance, u, process_cov, dt):
    """Performs the prediction steps of the EKF SLAM algorithm."""
    # Step 2: Construct F matrix
    Fx = getFx()

    # Step 3: Predict state
    expected_state = predictState(state, Fx, u, dt)

    # Step 4: 
    G_t = getG(state, Fx, u, dt)

    # Step 5: Expected state covariance
    expected_state_cov = predictCovariance(G_t, state_covariance, Fx, process_cov)

    return expected_state, expected_state_cov
