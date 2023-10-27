import numpy as np

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
