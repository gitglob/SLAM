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

    # Fake time
    time = np.linspace(0, 100, 1000)

    # Initialize state
    state = np.zeros((4, 1))

    # Initialize state covariance
    state_cov = np.eye((4)) * 0.01

    # Convert from moment to canonical form
    inf_matrix, inf_vector = moment2canonical(state_cov, state)

    # Initialize process noise
    process_cov = np.eye((4)) * 0.01

    # Initialize measurement noise
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
        expected_inf_matrix, expected_inf_vector = predict(inf_matrix, inf_vector, u, process_cov, dt)
        state_cov, state = canonical2moment(expected_inf_matrix, expected_inf_vector)

        # Steps 4-6: Correction
        inf_matrix, inf_vector = correct(expected_inf_matrix, expected_inf_vector, z, measurement_cov)
        state_cov, state = canonical2moment(inf_matrix, inf_vector)

    print(f"# of iterations: {i}")
    print("IF finished!")

if __name__ == "__main__":
    main()