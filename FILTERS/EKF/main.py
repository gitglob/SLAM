# Standard
# External
import numpy as np
# Local
from simulation.simulate_observations import simulate_sensors, simulate_spiral_movement
from simulation import range_noise_std, yaw_noise_std, random_seed
from visualization.plot_filter_results import plot_filter_trajectories
from .prediction import predict
from .correction import correct
from utils import polar2xy

np.random.seed(random_seed)

def get_Q_t(sigma_r=range_noise_std, sigma_phi=yaw_noise_std):
    """Returns the uncertainty matrix of the sensors."""
    Q_t = [[sigma_r**2,              0], 
           [0,            sigma_phi**2]]
    Q_t = np.array(Q_t).reshape(2,2)

    return Q_t

def main():
    # Step 1, start the Kalman Filter and initialize the covariance matrices:
    # - Process noise
    # - Measurement noise

    # Simulation - Spiral Trajectory ground truth
    x, y, theta, time, v, omega = simulate_spiral_movement()
    gt_states = np.vstack((x, y, theta)).T

    # Simulation - Spiral Trajectory sensor readings
    (sensor_measurements, sensor_ts) = simulate_sensors(x, y, time)

    # Initialize state
    state = np.array([x[0], y[0], theta[0]]).reshape((3,1)) # x, y, θ

    # Initialize state covariance
    state_cov = np.eye(3) * 1e-12
    
    # Initialize process noise
    process_cov = np.eye(3)*0.1

    # Initialize measurement noise
    measurement_cov = get_Q_t()

    # Keep track of all the states
    all_states = np.array(state).reshape(1,3,1)

    # Keep track of the prediction states
    prediction_states = np.empty((1,3,1))

    # Keep track of the correction states
    correction_states = np.empty((1,3,1))

    # Correction counters
    correction_counter = 0
    # Iterate over time
    for i, t in enumerate(time):
        print(f"Iteration: {i}, time: {t}")
        # Calculate dt
        if i == 0:
            continue
        else:
            dt = t - time[i-1]

        # Generate random control input (linear/rotational velocity)
        u = np.hstack((v[i], omega[i])).reshape((2,1))

        # Steps 2-3: Prediction
        expected_state, expected_state_cov = predict(state, state_cov, u, process_cov, dt)
        prediction_states = np.concatenate((prediction_states, expected_state[np.newaxis, :, :]), axis=0)
        state, state_cov = expected_state, expected_state_cov

        # Check if a correction is available
        if i < len(time) - 1 and correction_counter < len(sensor_ts) and sensor_ts[correction_counter] < time[i + 1]:
            # Get range measurement
            z = sensor_measurements[correction_counter].reshape((2,1))
            correction_counter += 1

            # Steps 4-7: Correction
            state, state_cov = correct(state, state_cov, z, measurement_cov)
            correction_states = np.concatenate((correction_states, state[np.newaxis, :, :]), axis=0)

        # Keep track of the all_states
        all_states = np.concatenate((all_states, state[np.newaxis, :, :]), axis=0)

    # Plot the results
    plot_filter_trajectories(all_states, 
                             prediction_states, correction_states, 
                             gt_states, "EKF")

    print(f"# of iterations: {i}")
    print(f"# of corrections: {correction_counter}")
    print("EKF finished!")

if __name__ == "__main__":
    main()