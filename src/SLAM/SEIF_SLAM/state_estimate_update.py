# Standard
# External
import numpy as np
# Local


def max_probability_state(inf_matrix, inf_vector):
    """
    Compute the state that maximizes the Gaussian probability distribution in its canonical form.

    The canonical form of the Gaussian distribution is:
    p(x) ∝ exp(-0.5 * x^T * Omega * x + xi^T * x)
    Where:
    - x is the state.
    - Omega is the information matrix (inverse of the covariance matrix).
    - xi is the information vector.

    Instead of directly inverting the information matrix (which can be computationally expensive),
    this function utilizes a numerical solver to efficiently compute the maximum probability state.

    Parameters:
    - inf_matrix (numpy.ndarray): The information matrix (2D array) of shape (n, n) where n is the length of the state.
    - inf_vector (numpy.ndarray): The information vector (1D array) of shape (n,).

    Returns:
    - numpy.ndarray: The state (1D array) that maximizes the Gaussian probability distribution in its canonical form.
    
    Example:
    --------
    inf_matrix = np.array([[4, 0], [0, 4]])
    inf_vector = np.array([2, 2])
    state = max_probability_state(inf_matrix, inf_vector)
    print(state)  # Outputs: [0.5 0.5]
    """
    
    # Using a numerical solver to find the state that maximizes the probability function
    state = np.linalg.solve(inf_matrix, inf_vector)
    
    return state

def update_state_estimate(inf_matrix, inf_vector):
    state_estimate = max_probability_state(inf_matrix, inf_vector)
    return state_estimate