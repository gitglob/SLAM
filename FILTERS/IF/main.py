# Standard
# External
import numpy as np
# Local
from .prediction import predict
from .correction import correct

def get_Q_t(sigma_r=100, sigma_phi=100):
    """Returns the uncertainty matrix of the sensors"""
    Q_t = [[sigma_r**2, 0], 
           [0,          sigma_phi**2]]
    Q_t = np.array(Q_t).reshape(2,2)

    return Q_t

def main():
    # Step 1, start the Kalman Filter and initialize the covariance matrices:
    # - Process noise
    # - Measurement noise

    # Fake time
    time = np.linspace(0, 1, 11)

    # Initialize state covariance
    state_cov = np.eye(4) * 1e-6
    inf_matrix = np.linalg.inv(state_cov) # Convert to canonical form

    # Initialize state
    state = np.zeros((4, 1))
    inf_vector = inf_matrix @ state # Convert to canonical form
    
    # Initialize process noise
    process_cov = np.eye(4) * 1e-6

    # Measurement noise initialization
    measurement_cov = get_Q_t()

    # Iterate over time
    for i, t in enumerate(time):
        print(f"Iteration: {i}")
        # Calculate dt
        if i == 0:
            dt = 0
        else:
            dt = t - time[i-1]

        # Generate random control input (linear/rotational velocity)
        v = np.random.random()
        omega = np.random.random()
        u = np.array([v, omega]).reshape((2,1))

        # Generate random measurement
        z = np.random.rand(2,1)

        # Steps 2-3: Prediction
        expected_inf_matrix, expected_inf_vector = predict(inf_matrix, inf_vector, u, process_cov, dt)
        
        # Steps 4-6: Correction
        inf_matrix, inf_vector = correct(expected_inf_matrix, expected_inf_vector, z, measurement_cov)

    print("IF finished!")

if __name__ == "__main__":
    main()