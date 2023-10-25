# Standard
# External
import numpy as np
# Local
from src.simulation.simulate_observations import simulate_sensors, simulate_spiral_movement
from src.simulation import range_noise_std, yaw_noise_std, random_seed
from src.visualization.plot_filter_results import plot_filter_trajectories
from .prediction import predict
from .correction import correct
from .utils import moment2canonical, canonical2moment

np.random.seed(random_seed)

def getQ(sigma_r=range_noise_std, sigma_phi=yaw_noise_std):
    """Returns the uncertainty matrix of the sensors."""
    Q_t = [[sigma_r**2,              0], 
           [0,            sigma_phi**2]]
    Q_t = np.array(Q_t).reshape(2,2)

    return Q_t

def main():
    # Step 1, start the Information Filter and initialize the covariance matrices:
    # - Process noise
    # - Measurement noise

    # Simulation - Spiral Trajectory ground truth
    x, y, theta, time, v, omega = simulate_spiral_movement()
    gt_states = np.column_stack((x, y, theta))

    # Simulation - Spiral Trajectory sensor readings
    (sensor_measurements, sensor_ts) = simulate_sensors(x, y, time)

    # Initialize state
    state = np.vstack((x[0], y[0], theta[0])) # x, y, θ

    # Initialize state covariance
    state_cov = np.eye(3) * 1e-12

    # Convert from moment to canonical form
    inf_matrix, inf_vector = moment2canonical(state_cov, state)

    # Initialize process noise
    process_cov = np.eye(3)*0.1

    # Initialize measurement noise
    measurement_cov = getQ()

    # Keep track of all the states
    all_states = np.zeros((len(time),3,1))
    all_states[0] = np.array(state)

    # Keep track of the prediction states
    prediction_states = np.zeros((len(time),3,1))

    # Keep track of the correction states
    correction_states = np.zeros((len(sensor_ts),3,1))

    # Correction counters
    correction_counter = 0

    # Iterate over time
    for i, t in enumerate(time):
        if i%100 == 0:
            print(f"Iteration: {i}, time: {t}")

        # Calculate dt
        if i == 0:
            continue
        else:
            dt = t - time[i-1]

        # Control input (linear/rotational velocity)
        u = np.vstack((v[i], omega[i]))

        # Steps 2-3: Prediction
        expected_inf_matrix, expected_inf_vector = predict(inf_matrix, inf_vector, u, process_cov, dt)
        state_cov, state = canonical2moment(expected_inf_matrix, expected_inf_vector)

        # Check if a correction is available
        if i < len(time) - 1 and correction_counter < len(sensor_ts) and sensor_ts[correction_counter] < time[i + 1]:
            # Get range measurement
            z = sensor_measurements[correction_counter].reshape((2,1))
            correction_counter += 1

            # Steps 4-6: Correction
            inf_matrix, inf_vector = correct(expected_inf_matrix, expected_inf_vector, z, measurement_cov)
            state_cov, state = canonical2moment(inf_matrix, inf_vector)
            correction_states[correction_counter-1] = state

    # Plot the results
    plot_filter_trajectories(all_states, 
                             prediction_states, correction_states, 
                             gt_states, "IF")

    print(f"# of iterations: {i}")
    print(f"# of corrections: {correction_counter}")
    print("IF finished!")

if __name__ == "__main__":
    main()