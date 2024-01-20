# Standard
import os
import time
# External
import numpy as np
# Local
from src.utils import read_world, read_data
from src.visualization.plot_slam_state import plot_slam_state
from .motion_update import update_motion
from .measurement_update import update_measurement
from .state_estimate_update import update_state_estimate
from .sparsification import sparsify, simple_sparsify
from .utils import moment2canonical, canonical2moment, detect_landmarks


def main():
    # Read world and sensor data
    world_path = os.path.join('exercises', '06_ekf_slam_framework', 'data', "world.dat")
    landmarks = read_world(world_path)
    NUM_LANDMARKS = len(landmarks["ids"])
    sensor_path = os.path.join('exercises', '06_ekf_slam_framework', 'data', "sensor_data.dat")
    data = read_data(sensor_path)

    # Initialize state
    state = np.zeros((3 + NUM_LANDMARKS*2, 1))

    # Initialize state covariance
    state_cov = np.eye((3 + NUM_LANDMARKS*2))*1e-6
    state_cov[3:, 3:] = np.eye(NUM_LANDMARKS*2) * 1000

    # Convert from moment to canonical form
    inf_matrix, inf_vector = moment2canonical(state_cov, state)
    prev_inf_matrix = inf_matrix
    
    # Initialize process noise
    process_cov = np.zeros((3,3))
    process_cov[0,0] = 0.1
    process_cov[1,1] = 0.1
    process_cov[2,2] = 0.01

    # Measurement noise
    measurement_cov = np.eye(2)*0.01
    
    # Initialize observed features
    map = []

    # Iterate over time
    predict_times = []
    correct_times = []
    for i, t in enumerate(data["timesteps"]):
        if i==20:
            print(f"Iteration: {i}, time: {t}")
            break

        # Extract velocity profile from the odometry readings
        dtheta1 = data["odometry"][i][0]
        dr = data["odometry"][i][1]
        dtheta2 = data["odometry"][i][2]
        displacement = np.vstack([dtheta1, dr, dtheta2])

        # Step 1: Motion Update
        start_time = time.time()
        expected_inf_vector, expected_inf_matrix, expected_state = update_motion(inf_vector, inf_matrix, state, displacement, process_cov, NUM_LANDMARKS)
        predict_times.append(time.time() - start_time)

        # Step 2: State Estimate Update
        state = update_state_estimate(expected_inf_matrix, expected_inf_vector, expected_state, NUM_LANDMARKS)

        # Get the current landmark observations
        observed_landmarks = data["sensor"][i]

        # # Step 3: Measurement Update
        start_time = time.time()
        inf_vector, inf_matrix, map = update_measurement(expected_inf_vector, expected_inf_matrix, state, observed_landmarks, measurement_cov, map, NUM_LANDMARKS)

        # # Step 3.5: Detect passive, new active, and active landmarks
        m_passive_idx, m_new_active_idx, m_active_idx = detect_landmarks(prev_inf_matrix, inf_matrix)
        prev_inf_matrix = inf_matrix

        # # Step 4: Sparsification
        inf_vector, inf_matrix = sparsify(inf_vector, inf_matrix, state, m_new_active_idx, m_active_idx, NUM_LANDMARKS)
        # inf_vector, inf_matrix = simple_sparsify(inf_vector, inf_matrix)
        correct_times.append(time.time() - start_time)

        # # Plot robot state
        state_cov, state = canonical2moment(inf_matrix, inf_vector)
        plot_slam_state(state, state_cov, t, landmarks, map, observed_landmarks, "SEIF")

    print(f"Avg. prediction time: {sum(predict_times)/len(predict_times):.4f} sec.")
    print(f"Avg. correction + sparsification time: {sum(correct_times)/len(correct_times):.4f} sec.")
    print(f"Avg. step time: {(sum(predict_times) + sum(correct_times))/(len(predict_times) + len(correct_times)):.4f} sec.")

    print("SEIF-SLAM finished!")

    print(f"Current state vector: \n{state}")
    # print(f"Current state covariance: {state_cov}")

if __name__ == "__main__":
    main()