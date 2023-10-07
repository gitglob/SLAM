# External
import numpy as np
# Local
from .prediction import predict
from .correction import correct
from . import NUM_LANDMARKS

def main():
    # Fake time
    time = np.linspace(0, 1, 11)

    # Initialize state
    state = np.zeros((3 + NUM_LANDMARKS*2, 1))

    # Initialize state covariance
    state_cov = np.zeros((3 + NUM_LANDMARKS*2, 3 + NUM_LANDMARKS*2))
    
    # Initialize process noise
    process_noise = np.zeros((3 + NUM_LANDMARKS*2, 3 + NUM_LANDMARKS*2))

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

        # Steps 1-5: Prediction
        expected_state, expected_state_cov = predict(state, state_cov, u, process_noise, dt)
        
        # Steps 6-23: Correction
        state, state_cov = correct(expected_state, expected_state_cov)

if __name__ == "__main__":
    main()