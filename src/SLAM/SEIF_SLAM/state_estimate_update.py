# Standard
# External
import numpy as np
from random import randint, seed
# Local
from src.simulation import random_seed
from src.utils import normalize_angle

seed(random_seed)


def getFx(NUM_LANDMARKS):
    """Calculates the Fx matrix, which is essential to apply the motion model to the pose state, and not the landmarks.
    Fx maps the robot's state from a low dimensional space, to a high dimensional space."""

    Fx = np.zeros((3, 3 + 2*NUM_LANDMARKS))
    Fx[:3, :3] = np.eye(3)

    return Fx

def getF_i(NUM_LANDMARKS, i):
    Fi = np.zeros((2, 2*(NUM_LANDMARKS-i) + 2 + 2*(i-1)))
    Fi[:, 2*(NUM_LANDMARKS - i) : 2*(NUM_LANDMARKS - i) + 2] = np.eye(2)

    return Fi

def updateState(F, inf_matrix, inf_vector, expected_state):
    term1 = np.linalg.inv(F@inf_matrix@F.T)
    term2 = F@(inf_vector - inf_matrix@expected_state + inf_matrix@F.T@F@expected_state)

    state = term1@term2

    return state

def update_state_estimate(inf_matrix, inf_vector, expected_state, NUM_LANDMARKS):
    state = np.zeros((3 + NUM_LANDMARKS*2, 1))

    # Step 1: For a small set of map features
    feature_subset = [randint(1, NUM_LANDMARKS) for _ in range(int(NUM_LANDMARKS/3))]
    for i in feature_subset:
        # Step 2
        Fi = getF_i(NUM_LANDMARKS, i)

        # Step 3
        j = i-1
        state[3+j : 3+j+2] = updateState(Fi, inf_matrix[3:, 3:], inf_vector[3:], expected_state[3:])

    # Step 4: For all other map features
    all_features = list(range(1, 10))
    remaining_features = [num for num in all_features if num not in feature_subset]
    for i in remaining_features:
        # Step 5: Keep the expected state
        j = i-1
        state[3+j : 3+j+2] = expected_state[3+j : 3+j+2]

    # Step 6
    Fx = getFx(NUM_LANDMARKS)

    # Step 7
    state[:3] = updateState(Fx, inf_matrix, inf_vector, expected_state)
    state[2] = normalize_angle(state[2])

    # Step 8
    return state
