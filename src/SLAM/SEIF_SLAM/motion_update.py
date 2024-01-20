# External
import numpy as np
from numpy.linalg import inv
# Local
from src.utils import velocityModel, normalize_angle


# Step 1
def getFx(NUM_LANDMARKS):
    """Calculates the Fx matrix, which is essential to apply the motion model to the pose state, and not the landmarks.
    Fx maps the robot's state from a low dimensional space, to a high dimensional space."""

    pose_Fx = np.eye(3)
    landmark_Fx = np.zeros((3, 2*NUM_LANDMARKS))

    Fx = np.hstack((pose_Fx, landmark_Fx))

    return Fx

# Step 2
def get_delta(state, displacement):
    """Gets the displacement based on the motion model."""

    # Extract linear and rotational displacement
    dtheta1, dr, dtheta2 = displacement
    
    # Extract heading
    theta = state[2]

    # Displacement from the velocity model - circular arc model
    dx = dr * np.cos(theta + dtheta1)
    dy = dr * np.sin(theta + dtheta1)
    dtheta = dtheta1 + dtheta2
    delta = np.vstack((dx, dy, dtheta))

    return delta

# Step 3
def getDelta(state, displacement):
    """Calculates the Jacobian of the motion model at the given state under the current control input."""

    # Extract linear and rotational displacement
    dtheta1, dr, _ = displacement

    # Extract heading
    theta = state[2]

    # Initialize the Jacobian with a zero matrix
    Delta = np.zeros((3, 3))

    # Jacobian of the motion model
    Delta[0,2] = -dr * np.sin(theta + dtheta1)
    Delta[1,2] = dr * np.cos(theta + dtheta1)

    return Delta

# Step 4
def getPsi(Fx, Delta):
    I = np.eye(3)
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

    return kappa

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
    expected_state[2] = normalize_angle(expected_state[2])

    return expected_state

# Motion Update
def update_motion(inf_vector, inf_matrix, state, displacement, process_cov, NUM_LANDMARKS):
    Fx = getFx(NUM_LANDMARKS)
    delta = get_delta(state, displacement)
    Delta = getDelta(state, displacement)
    Psi = getPsi(Fx, Delta)
    lamda = getLamda(Psi, inf_matrix)
    Phi = getPhi(inf_matrix, lamda)
    kappa = getKappa(Phi, Fx, process_cov)
    expected_inf_matrix = predictInfMatrix(Phi, kappa)
    expected_inf_vector = predictInfVector(inf_vector, lamda, kappa, state, expected_inf_matrix, Fx, delta)
    expected_state = predictState(state, Fx, delta)

    return expected_inf_vector, expected_inf_matrix, expected_state
