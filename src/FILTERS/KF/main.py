# Standard
# External
import numpy as np
# Local
from .prediction import predict
from .correction import correct

def getQ(sigma_r=100, sigma_phi=100):
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
    time = np.linspace(0, 100, 1000)

    # Initialize state
    state = np.zeros((4, 1))

    # Initialize state covariance
    state_cov = np.eye((4)) * 0.01
    
    # Initialize process noise
    process_cov = np.eye((4)) * 0.01

    # Measurement noise initialization
    measurement_cov = getQ()

    # Iterate over time
    for i, t in enumerate(time):
        if i%100 == 0:
            print(f"Iteration: {i}, time: {t}")

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
        expected_state, expected_state_cov = predict(state, state_cov, u, process_cov, dt)
        
        # Steps 4-7: Correction
        state, state_cov = correct(expected_state, expected_state_cov, z, measurement_cov)

    print(f"# of iterations: {i}")
    print("IF finished!")
    
if __name__ == "__main__":
    main()