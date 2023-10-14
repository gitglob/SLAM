# Standard
# External
import numpy as np
# Local
from .prediction import predict
from .correction import correct
from .utils import getLamda
from . import alpha, beta, kappa

def get_Q_t(sigma_r=100, sigma_phi=100):
    """
    Returns the uncertainty matrix of the sensors.
    
    We assume that we have a sensor that measures the range and the bearing of the robot, for simplicity.
    """
    Q_t = [[sigma_r**2,              0], 
           [0,            sigma_phi**2]]
    Q_t = np.array(Q_t).reshape(2,2)

    return Q_t

def main():
    # Step 1, start the Kalman Filter and initialize the covariance matrices:
    # - Process noise
    # - Measurement noise

    # Fake time
    time = np.linspace(0, 1, 11)

    # Initialize state
    state = np.zeros((3, 1)) # x, y, θ

    # Initialize state covariance
    state_cov = np.zeros((3, 3))
    
    # Initialize process noise
    process_cov = np.zeros((3, 3))

    # Measurement noise initialization
    measurement_cov = get_Q_t()

    # Get the UKF parameters
    n = state.shape[0]
    lamda = getLamda(alpha, n, kappa)

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
        u = np.hstack((v, omega)).reshape((2,1))

        # Generate random measurement
        z = np.random.rand(2,1)

        # Steps 2-3: Prediction
        expected_state, expected_state_cov = predict(state, state_cov, u, process_cov, n, lamda, dt)
        
        # Steps 4-7: Correction
        state, state_cov = correct(expected_state, expected_state_cov, z, measurement_cov, n, lamda)

    print("UKF finished!")

if __name__ == "__main__":
    main()