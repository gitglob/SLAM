# Standard
import os
import time
# External
import numpy as np
# Local
from src.utils import read_world, read_data
from src.visualization.plot_slam_state import plot_slam_state
from .prediction import predict
from .correction import correct

def getFx(NUM_LANDMARKS):
    """Calculates the Fx matrix, which is essential to apply the motion model to the pose state, and not the landmarks.
    Fx maps the robot's state from a low dimensional space, to a high dimensional space."""

    Fx = np.zeros((3, 3 + 2*NUM_LANDMARKS))
    Fx[:3, :3] = np.eye(3)

    return Fx

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
    state_cov = np.zeros((3 + NUM_LANDMARKS*2, 3 + NUM_LANDMARKS*2))
    state_cov[3:, 3:] = np.eye(NUM_LANDMARKS*2) * 1000
    
    # Initialize process noise
    process_cov = np.zeros((3,3))
    process_cov[0,0] = 0.1
    process_cov[1,1] = 0.1
    process_cov[2,2] = 0.01
    
    # Initialize Fx
    Fx = getFx(NUM_LANDMARKS)
    
    # Initialize observed features
    map = []

    # Iterate over time
    predict_times = []
    correct_times = []
    for i, t in enumerate(data["timesteps"]):
        if i%100 == 0:
            print(f"Iteration: {i}, time: {t}")
            
        # Extract velocity profile from the odometry readings
        dtheta1 = data["odometry"][i][0]
        dr = data["odometry"][i][1]
        dtheta2 = data["odometry"][i][2]
        displacement = np.array([dtheta1, dr, dtheta2]).reshape((3,1))

        # Steps 1-5: Prediction
        start_time = time.time()
        expected_state, expected_state_cov = predict(state, state_cov, displacement, process_cov, Fx, NUM_LANDMARKS)
        predict_times.append(time.time() - start_time)

        # Get the current landmark observations
        observed_landmarks = data["sensor"][i]

        # Steps 6-23: Correction
        start_time = time.time()
        state, state_cov, map = correct(expected_state, expected_state_cov, NUM_LANDMARKS, observed_landmarks, map)
        correct_times.append(time.time() - start_time)

        # Plot robot state
        plot_slam_state(state, state_cov, t, landmarks, map, observed_landmarks, "EKF")

    print(f"Avg. prediction time: {sum(predict_times)/len(predict_times):.4f} sec.")
    print(f"Avg. correction time: {sum(correct_times)/len(correct_times):.4f} sec.")
    print(f"Avg. step time: {(sum(predict_times) + sum(correct_times))/(len(predict_times) + len(correct_times)):.4f} sec.")

    print("EKF-SLAM finished!")

    print(f"Current state vector: \n{state}")
    # print(f"Current state covariance: {state_cov}")

if __name__ == "__main__":
    main()