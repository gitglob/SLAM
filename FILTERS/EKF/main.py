# Standard
# External
import numpy as np
# Local
from .prediction import predict
from .correction import correct
from simulation.simulate_observations import simulate_sensors, simulate_spiral_movement
from visualization.plot_filter_results import plot_filter_trajectories


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

    # Simulation - Spiral Trajectory ground truth
    x, y, theta, time = simulate_spiral_movement()
    gt_states = np.vstack((x, y, theta)).T

    # Simulation - Spiral Trajectory sensor readings
    (ranges, range_ts), (yaws, yaw_ts) = simulate_sensors(x, y, time)

    # Initialize state
    state = np.array([x[0], y[0], theta[0]]).reshape((3,1)) # x, y, θ
    all_states = np.array(state).reshape(1,3,1)

    # Initialize state covariance
    state_cov = np.zeros((3, 3))
    
    # Initialize process noise
    process_cov = np.zeros((3, 3))

    # Measurement noise initialization
    measurement_cov = get_Q_t()

    # Correction counters
    ct_range = 0
    ct_yaw = 0
    # Iterate over time
    for i, t in enumerate(time):
        print(f"Iteration: {i}, time: {t}")
        # Calculate dt
        if i == 0:
            dt = 0
        else:
            dt = t - time[i-1]

        # Generate random control input (linear/rotational velocity)
        v = np.random.random()
        omega = np.random.random()
        u = np.hstack((v, omega)).reshape((2,1))

        # Steps 2-3: Prediction
        expected_state, expected_state_cov = predict(state, state_cov, u, process_cov, dt)
        
        # Check if a correction is available
        if i < len(time) - 1 and ct_range < len(range_ts) and range_ts[ct_range] < time[i + 1]:
            # Get range measurement
            z = np.vstack((ranges[ct_range], state[2])).reshape((2,1))
            ct_range += 1

            # Steps 4-7: Correction
            state, state_cov = correct(expected_state, expected_state_cov, z, measurement_cov)

        elif i < len(time) - 1 and ct_range < len(yaw_ts) and yaw_ts[ct_yaw] < time[i + 1]:
            # Get yaw measurement
            z = np.vstack((state[0]**2 + state[1]**2, yaws[ct_yaw])).reshape((2,1))
            ct_range += 1

            # Steps 4-7: Correction
            state, state_cov = correct(expected_state, expected_state_cov, z, measurement_cov)

        # Keep track of the all_states
        all_states = np.concatenate((all_states, state[np.newaxis, :, :]), axis=0)

    # Plot the results
    plot_filter_trajectories(all_states, time, gt_states, time, "EKF")

    print("EKF finished!")

if __name__ == "__main__":
    main()