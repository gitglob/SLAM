# External
import numpy as np
from numpy.linalg import inv
# Local
from utils import velocityModel
from . import NUM_LANDMARKS


# Step 1
def getFx():
    """Calculates the Fx matrix, which is essential to apply the motion model to the pose state, and not the landmarks.
    Fx maps the robot's state from a low dimensional space, to a high dimensional space."""

    pose_Fx = np.eye(3)
    landmark_Fx = np.zeros((3, 2*NUM_LANDMARKS))

    Fx = np.hstack((pose_Fx, landmark_Fx))

    return Fx

# Step 2
def get_delta(state, u, dt):
    """Gets the displacement based on the motion model."""

    # Displacement from the velocity model - circular arc model
    delta = velocityModel(state, u, dt)

    return delta

# Step 3
def getDelta(state, u, dt):
    """Calculates the Jacobian of the motion model at the given state under the current control input."""

    # Extract linear velocity and yaw rate
    v, omega = u

    # Extract heading
    theta = state[2]

    # Initialize the Jacobian with a zero matrix
    Delta = np.zeros((3, 3))

    # Handle the case when omega is zero (or close to zero) - straight line motion
    epsilon = 1e-6
    if omega < epsilon:
        Delta[0, 2] = -v * np.sin(theta) * dt
        Delta[1, 2] = v * np.cos(theta) * dt
    else:
        # Jacobian elements for the general motion model (circular arc model)
        Delta[0, 2] = (-v/omega) * np.cos(theta) + (v/omega) * np.cos(theta + omega*dt)
        Delta[1, 2] = (-v/omega) * np.sin(theta) + (v/omega) * np.sin(theta + omega*dt)

    return Delta

# Step 4
def getPsi(Fx, Delta):
    I = np.eye(3 + 2*NUM_LANDMARKS)
    Psi = Fx.T @ (inv(I+Delta) - I) @ Fx

    return Psi

# Step 5
def getLamda(Psi, inf_matrix):
    lamda = Psi.T @ inf_matrix + inf_matrix @ Psi + Psi.T @ inf_matrix @ Psi

    return lamda

# Step 6
def getPhi(inf_matrix, lamda):
    Phi = inf_matrix + lamda

    return Phi

# Step 7
def getKappa(Phi, Fx, process_cov):
    kappa = Phi@Fx.T @ inv(inv(process_cov) + Fx@Phi@Fx.T) @ Fx@Phi

# Step 8
def predictInfMatrix(Phi, kappa):
    expected_inf_matrix = Phi - kappa

    return expected_inf_matrix

# Step 9
def predictInfVector(inf_vector, lamda, kappa, state, expected_inf_matrix, Fx, delta):
    expected_inf_vector = inf_vector + (lamda-kappa)@state + expected_inf_matrix@Fx.T@delta

    return expected_inf_vector

# Step 10
def predictState(state, Fx, delta):
    expected_state = state + Fx.T@delta

    return expected_state

# Motion Update
def update_motion(inf_vector, inf_matrix, state, u, process_cov, dt):
    Fx = getFx()
    delta = get_delta(state, u, dt)
    Delta = getDelta(state, u, dt)
    Psi = getPsi(Fx, Delta)
    lamda = getLamda(Psi, inf_matrix)
    Phi = getPhi(inf_matrix, lamda)
    kappa = getKappa(Phi, Fx, process_cov)
    expected_inf_matrix = predictInfMatrix(Phi, kappa)
    expected_inf_vector = predictInfVector(inf_vector, lamda, kappa, state, expected_inf_matrix, Fx, delta)
    expected_state = predictState(state, Fx, delta)

    return expected_inf_matrix, expected_inf_vector, expected_state