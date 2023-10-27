# External
import numpy as np
# Local
from src.simulation import range_noise_std, yaw_noise_std, random_seed
from .motion_update import update_motion
from .measurement_update import update_measurement
from .state_estimate_update import update_state_estimate
from .sparsification import sparsify
from .utils import moment2canonical, canonical2moment
from . import NUM_LANDMARKS

# Step 1
def getQ(sigma_r=range_noise_std, sigma_phi=yaw_noise_std):
    """Returns the uncertainty matrix of the sensors."""
    Q_t = [[sigma_r**2,              0], 
           [0,            sigma_phi**2]]
    Q_t = np.array(Q_t).reshape(2,2)

    return Q_t

def main():
    # Fake time
    time = np.linspace(0, 1, 11)

    # Initialize state
    state = np.zeros((3 + NUM_LANDMARKS*2, 1))

    # Initialize state covariance
    state_cov = np.eye((3 + NUM_LANDMARKS*2)) * 1e-6
    
    # Convert from moment to canonical form
    inf_matrix, inf_vector = moment2canonical(state_cov, state)

    # Initialize process noise
    process_cov = np.eye((3 + NUM_LANDMARKS*2)) * 0.1

    # Initialize measurement noise
    measurement_cov = getQ()

    # Iterate over time
    for i, t in enumerate(time):
        # Calculate dt
        if i == 0:
            dt = 0
        else:
            dt = t - time[i-1]

        # Generate random control input (linear/rotational velocity)
        v = np.random.random()
        omega = np.random.random()
        u = np.array([v, omega]).reshape((2,1))

        # Step 1: Motion Update
        expected_inf_matrix, expected_inf_vector, expected_state = update_motion(inf_vector, inf_matrix, state, u, process_cov, dt)
        
        # Step 2: Measurement Update
        inf_vector, inf_matrix = update_measurement(expected_inf_vector, expected_inf_matrix, expected_state, observed_features, measurement_cov, seen_features, state_cov)

        # Step 3: State Estimate Update
        state_estimate = update_state_estimate(inf_matrix, inf_vector)

        # Step 4: Sparsification
        sparse_inf_vector, sparse_inf_matrix = sparsify(inf_vector, inf_matrix, state_estimate)

        # Step 5: Return the sparce information vector and matrix, and the state estimate
        inf_vector = sparse_inf_vector
        inf_matrix = sparse_inf_matrix
        state = state_estimate        

if __name__ == "__main__":
    main()