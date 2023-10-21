# Standard
# External
import numpy as np
# Local
from .prediction import predict
from .correction import correct
from visualization.plot_filter_results import plot_filter_trajectories
from simulation.simulate_observations import simulate_sensors, simulate_spiral_movement

def get_Q_t(sigma_x=0.1, sigma_phi=0.1):
    """Returns the uncertainty matrix of the sensors."""
    Q_t = [[sigma_x**2,              0], 
           [0,            sigma_phi**2]]
    Q_t = np.array(Q_t).reshape(2,2)

    return Q_t


def main():
    # Step 1, start the Kalman Filter and initialize the covariance matrices:
    # - Process noise
    # - Measurement noise

    # Fake time
    time = np.linspace(0, 1, 11)

    # Initialize state covariance
    ebs = 1e-6
    state_cov = np.eye(3) * ebs
    inf_matrix = np.linalg.inv(state_cov)

    # Initialize state
    state = np.zeros((3, 1))  + ebs
    inf_vector = inf_matrix @ state
    
    # Initialize process noise
    process_cov = np.eye(3) * ebs

    # Measurement noise initialization
    measurement_cov = get_Q_t()

    # Iterate over time
    for i, t in enumerate(time):
        if i%100 == 0:
            print(f"Iteration: {i}, time: {t}")

        # Calculate dt
        if i == 0:
            continue
        else:
            dt = t - time[i-1]

        # Generate random control input (linear/rotational velocity)
        v = np.random.random()
        omega = np.random.random()
        u = np.hstack((v, omega)).reshape((2,1))

        # Generate random measurement
        z = np.random.rand(2,1)

        # Steps 2-5: Prediction
        expected_inf_matrix, expected_inf_vector, expected_state = predict(inf_matrix, inf_vector, state, u, process_cov, dt)
        
        # Steps 6-8: Correction
        inf_matrix, inf_vector = correct(expected_inf_matrix, expected_inf_vector, expected_state, z, measurement_cov)

    # Plot the results
    breakpoint()
    plot_filter_trajectories(all_states, 
                             prediction_states, correction_states, 
                             gt_states, "UKF")
    
    print(f"# of iterations: {i}")
    print(f"# of corrections: {correction_counter}")
    print("EIF finished!")

if __name__ == "__main__":
    main()