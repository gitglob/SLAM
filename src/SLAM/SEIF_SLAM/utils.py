# Standard
# Externals
import numpy as np
import random
# Local
from . import NUM_LANDMARKS

def moment2canonical(state_cov, state):
    """Converts the state and state covariance from the moment/standard form
    to the canonical/information form."""

    inf_matrix = np.linalg.inv(state_cov)
    inf_vector = inf_matrix @ state

    return inf_matrix, inf_vector


def canonical2moment(inf_matrix, inf_vector):
    """Converts the state and state covariance from the moment/standard form
    to the canonical/information form."""

    state_cov = np.linalg.inv(inf_matrix)
    state = state_cov @ inf_vector

    return state_cov, state

def detect_landmarks(prev_inf_matrix, current_inf_matrix, threshold=1e-5):
    """
    Detect passive, active, and new-active landmarks from the current and previous information matrices.

    Parameters:
    - prev_inf_matrix (numpy.ndarray): The previous information matrix.
    - current_inf_matrix (numpy.ndarray): The current information matrix.
    - threshold (float): The minimum value in the information matrix to consider a landmark active.

    Returns:
    - tuple: Sets of indices representing active, passive, and new-active landmarks.
    """

    # Assuming the first 3 rows/columns of the information matrix are for robot pose
    active_landmarks = set()
    passive_landmarks = set()

    # Detect active and passive landmarks from current information matrix
    for i in range(NUM_LANDMARKS):
        idx = 3 + 2*i
        # If the absolute values in the matrix for this landmark are larger than the threshold
        # then the landmark is active
        if np.abs(current_inf_matrix[idx:idx+2, idx:idx+2]).sum() > threshold:
            active_landmarks.add(i)
        else:
            passive_landmarks.add(i)

    # Detect landmarks that were passive in the previous step but are active now
    new_active_landmarks = set()
    for i in range(NUM_LANDMARKS):
        idx = 3 + 2*i
        if i in active_landmarks and np.abs(prev_inf_matrix[idx:idx+2, idx:idx+2]).sum() <= threshold:
            new_active_landmarks.add(i)

    return passive_landmarks, new_active_landmarks, active_landmarks

def pseudo_observe_landmarks(NUM_LANDMARKS):
    """
    Get two unique random integers ranging from 0 to NUM_LANDMARKS-1.

    Parameters:
    - NUM_LANDMARKS (int): The upper limit for the random integers.

    Returns:
    - tuple: Two unique random integers.
    """

    return tuple(random.sample(range(NUM_LANDMARKS), 2))