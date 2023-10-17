# External
import numpy as np
# Local
from . import NUM_LANDMARKS


# Step 2
def getF_x():
    """Calculates the F_x matrix, which is essential to apply the motion model to the pose state, and not the landmarks.
    F_x maps the robot's state from a low dimensional space, to a high dimensional space."""

    pose_F_x = np.eye(3)
    landmark_F_x = np.zeros((3, 2*NUM_LANDMARKS))

    F_x = np.hstack((pose_F_x, landmark_F_x))

    return F_x

# Step 3
def predictState(state, F_x, u, dt):
    """Predicts the new pose μ of the robot, based on the circular arc velocity model."""

    # Extract linear velocity and yaw rate
    v, omega = u

    # Extract heading
    theta = state[2]

    # Displacement from the velocity model - circular arc model
    dx = (-v/omega) * np.sin(theta) + (v/omega) * np.sin(theta + omega*dt)
    dy = (v/omega) * np.cos(theta) - (v/omega) * np.cos(theta + omega*dt)
    dtheta = omega*dt
    velocity_model = np.array([dx, dy, dz]).reshape((3, 1))

    expected_state = state + F_x.T @ velocity_model

    return expected_state

# Step 4
def getG(state, F_x, u, dt):
    """Calculates the Jacobian of the motion model at the given state under the current control input."""

    # Extract linear velocity and yaw rate
    v, omega = u

    # Extract heading
    theta = state[2]

    # Jacobian of the motion model
    I = np.eye(2*NUM_LANDMARKS + 3, 2*NUM_LANDMARKS + 3) # Identity for the 2Nx2N landmark part
    G_t_x_deriv = (-v/omega) * np.cos(theta) + (v/omega) * np.cos(theta + omega*dt)
    G_t_y_deriv = (-v/omega) * np.sin(theta) - (v/omega) * np.sin(theta + omega*dt)
    velocity_model_deriv = np.vstack((G_t_x_deriv, G_t_y_deriv, 0))
    G_t_x = np.hstack((np.zeros((3,1)), np.zeros((3,1)), velocity_model_deriv)) # Jacobian of the motion

    G_t = I + F_x.T @ G_t_x @ F_x

    return G_t

# Step 5
def predictCovariance(G_t, state_covariance, F_x, process_cov):
    """Predicts the new covariance matrix."""

    # Extract the process noise that corresponds to the robot's pose
    R_t_x = process_cov[:3, :3]

    # Process noise - only affects the robot, not the landmarks
    R_t = F_x.T @ R_t_x @ F_x

    # In the new uncertainty, only the parts that contain the robot's pose will change
    updated_covariance = G_t @ state_covariance @ G_t.T

    expected_covariance = updated_covariance + R_t

    return expected_covariance

# Step 1: Prediction
def predict(state, state_covariance, u, process_cov, dt):
    """Performs the prediction steps of the EKF SLAM algorithm."""
    # Step 2: Construct F matrix
    F_x = getF_x()

    # Step 3: Predict state
    expected_state = predictState(state, F_x, u, dt)

    # Step 4: 
    G_t = getG(state, F_x, u, dt)

    # Step 5: Expected state covariance
    expected_state_cov = predictCovariance(G_t, state_covariance, F_x, process_cov)

    return expected_state, expected_state_cov
