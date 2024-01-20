# Standard
# External
import numpy as np
from numpy.linalg import inv
# Local


# Step 5
def initializeLandmarkPosition(expected_state, landmark_state):
    """Initializes the position of a landmark if it has not been seen before."""
    expected_state_landmark_x = expected_state[0] + landmark_state[0]*np.cos(landmark_state[1] + expected_state[2])
    expected_state_landmark_y = expected_state[1] + landmark_state[0]*np.sin(landmark_state[1] + expected_state[2])

    expected_landmark_state = np.vstack([expected_state_landmark_x, expected_state_landmark_y])

    return expected_landmark_state

# Step 7
def get_delta(state, j):
    """Gets the displacement based on the motion model."""
    delta = state[3 + 2*j : 3 + 2*j + 2] - state[:2]

    return delta

# Step 9
def predict_observation(q, delta, theta):
    "Predicts the landmark observation."
    dx = delta[0].item()
    dy = delta[1].item()
    theta = theta.item()
    expected_observation = np.vstack([np.sqrt(q), np.arctan2(dy, dx) - theta])

    return expected_observation

# Step 10
def getH(q, delta, j, NUM_LANDMARKS):
    """Returns the Jacobian of the expected observation function, for the specific landmark and the current robot's pose."""
    dx = delta[0].item()
    dy = delta[1].item()

    H = np.zeros((2, 3 + 2*NUM_LANDMARKS))
    H[ : , :3                    ] = np.array([[-np.sqrt(q)*dx, -np.sqrt(q)*dy,  0],
                                               [dy,             -dx,            -q]])
    H[ : , 3         : 3+2*j-2   ] = np.zeros((2, 2*j-2))
    H[ : , 3+2*j-2   : 3+2*j-2+2 ] = np.array([[np.sqrt(q)*dx, np.sqrt(q)*dy],
                                               [-dy,           dx           ]])
    H[ : , 3+2*j-2+2 :           ] = np.zeros((2, 2*NUM_LANDMARKS - 2*j))

    H = 1/q * H

    return H

# Step 12
def update_inf_vector(H, Q, z, expected_z, state):
    """Updates the information vector."""
    inf_vector = H.T@inv(Q) @ (z - expected_z + H@state)

    return inf_vector

# Step 13
def update_inf_matrix(H, Q):
    """Updates the information matrix."""
    inf_matrix = H.T@inv(Q)@H

    return inf_matrix

# Motion Update
def update_measurement(expected_inf_vector, expected_inf_matrix, state, observed_landmarks, Q, map, NUM_LANDMARKS):
    # Step 1 - get measurement covariance - Q

    # Step 2 - iterate over all observed features
    expected_inf_vector_sum = np.zeros_like(expected_inf_vector)
    expected_inf_matrix_sum = np.zeros_like(expected_inf_matrix)
    for i, feature in enumerate(observed_landmarks):
        # Step 3: Extract landmark id and position
        j = feature[0]
        z = np.vstack(feature[1:3])

        # Step 4
        if j not in map:
            # Step 5
            state[3 + 2*j : 3 + 2*j + 2] = initializeLandmarkPosition(state, z)

            # Keep track of seen features
            map.append(j)

            # Step 6 - endif

        # Step 7
        delta = get_delta(state, j)

        # Step 8
        q = (delta.T@delta).item()

        # Step 9
        expected_z = predict_observation(q, delta, state[2])

        # Step 10
        Hi = getH(q, delta, j+1, NUM_LANDMARKS)

        # Step 11
        expected_inf_vector_sum += update_inf_vector(Hi, Q, z, expected_z, state)

        # Step 12
        expected_inf_matrix_sum += update_inf_matrix(Hi, Q)

        # Step 13 - endfor

    inf_vector = expected_inf_vector + expected_inf_vector_sum
    inf_matrix = expected_inf_matrix + expected_inf_matrix_sum

    # Step 14
    return inf_vector, inf_matrix, map
