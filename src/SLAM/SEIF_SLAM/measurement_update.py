# Standard
# External
import numpy as np
from numpy.linalg import inv
# Local
from . import NUM_LANDMARKS

# Step 5
def initializeLandmarkPosition(expected_robot_state, feature_state):
    """Initializes the position of a landmark if it has not been seen before."""
    expected_state_landmark_x = expected_robot_state[0] + feature_state[0]*np.cos(feature_state[1] + expected_robot_state[2])
    expected_state_landmark_y = expected_robot_state[1] + feature_state[0]*np.sin(feature_state[1] + expected_robot_state[2])

    expected_landmark_state = np.array([expected_state_landmark_x, expected_state_landmark_y]).reshape(2,1)

    return expected_landmark_state

# Step 7
def get_delta(expected_robot_state, expected_landmark_state):
    """Gets the displacement based on the motion model."""
    delta = expected_landmark_state - expected_robot_state

    return delta

# Step 9
def predict_observation(q, delta, expected_state):
    dx = delta[0].item()
    dy = delta[1].item()
    theta = expected_state[2]
    expected_observation = np.array([np.sqrt(q), np.arctan2(dy, dx) - theta])

    return expected_observation

# Step 10
def getH(q, delta, j):
    """Returns the Jacobian of the expected observation function, for the specific landmark and the current robot's pose."""
    dx = delta[0].item()
    dy = delta[1].item()

    H = np.zeros(2, 3 + 2*NUM_LANDMARKS)
    H[:, :3] = np.array([[-np.sqrt(q)*dx, -np.sqrt(q)*dy,  0],
                          [dy,             -dx,            -q]])
    H[:, 3 : 3+2*j-2] = np.zeros((2, 2*j-2))
    H[:, 3+2*j-2 : 3+2*j-2+2] = np.array([[np.sqrt(q)*dx, np.sqrt(q)*dy],
                                          [-dy,           dx]])
    H[:, 3+2*j-2+2:] = np.zeros((2, 2*NUM_LANDMARKS - 2*j))

    H = 1/q * H

    return H

# Step 12
def update_inf_vector(expected_inf_vector, state_cov, H, Q, z, expected_z, expected_state):
    inf_vector = expected_inf_vector + state_cov@H.T@inv(Q) @ (z - expected_z + H@expected_state)

    return inf_vector

# Step 13
def update_inf_matrix(expected_inf_matrix, state_cov, H, Q):
    inf_matrix = expected_inf_matrix + state_cov@H.T@inv(Q)@H

    return inf_matrix

# Motion Update
def update_measurement(expected_inf_vector, expected_inf_matrix, expected_state, observed_features, measurement_cov, all_seen_features, state_cov):
    # Step 1 - get measurement covariance\
    Q = measurement_cov

    # Step 2 - iterate over all observed features
    observed_features = np.array([[0.1,0.1], [1,1], [2,2], [3,3], [4,4]]) # observed features
    for i, z in enumerate(observed_features):
        # Step 3
        j = i

        # Step 4
        if z not in all_seen_features:
            # Step 5
            expected_landmark_state = initializeLandmarkPosition(expected_state, z)

            # Keep track of seen features
            all_seen_features.append(z)

            # Step 6 - endif

        # Step 7
        delta = get_delta(expected_state, expected_landmark_state)

        # Step 8
        q = delta.T@delta

        # Step 9
        expected_observation = predict_observation(q, delta, expected_state)

        # Step 10
        H = getH(q, delta, j)

        # Step 11 - endfor

    # Step 12
    inf_vector = update_inf_vector(expected_inf_vector, state_cov, H, Q, z, expected_observation, expected_state)

    # Step 13
    inf_matrix = update_inf_matrix(expected_inf_matrix, state_cov, H, Q)

    # Step 14
    return inf_vector, inf_matrix, all_seen_features
