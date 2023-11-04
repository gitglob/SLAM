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


def getQ(num_landmarks=1):
    """Returns the measurement uncertainty matrix for {num_landmarks} landmarks."""
    Q = np.eye(2*num_landmarks) * 0.01

    return Q

def main():
    # Read world and sensor data
    world_path = os.path.join('exercises', '09_ukf_slam_framework', 'data', "world.dat")
    landmarks = read_world(world_path)
    sensor_path = os.path.join('exercises', '09_ukf_slam_framework', 'data', "sensor_data.dat")
    data = read_data(sensor_path)

    # Initialize state
    state = np.zeros((3, 1))

    # Initialize state covariance
    state_cov = np.eye(3) * 0.001
    
    # Initialize process noise
    process_cov = np.zeros((3,3))
    process_cov[0,0] = 0.1
    process_cov[1,1] = 0.1
    process_cov[2,2] = 0.01

    # Initialize sensor noise
    measurement_cov = getQ()
        
    # Initialize observed features
    map = []
    
    # Get the UKF parameters
    gamma = 3

    # Iterate over time
    predict_times = []
    correct_times = []
    for i, t in enumerate(data["timesteps"]):
        if i%100 == 0:
            print(f"Iteration: {i}, time: {t}")
        if i==30:
            break
            
        # Extract velocity profile from the odometry readings
        dtheta1 = data["odometry"][i][0]
        dr = data["odometry"][i][1]
        dtheta2 = data["odometry"][i][2]
        displacement = np.array([dtheta1, dr, dtheta2]).reshape((3,1))
           
        # Steps 1-5: Prediction
        start_time = time.time()
        state, state_cov = predict(state, state_cov, displacement, process_cov, gamma)
        predict_times.append(time.time() - start_time)

        # Get the current landmark observations
        observed_landmarks = data["sensor"][i]

        # Steps 6-21: Correction
        start_time = time.time()
        state, state_cov, map = correct(state, state_cov, measurement_cov, observed_landmarks, map, gamma)
        correct_times.append(time.time() - start_time)

        # Plot robot state
        plot_slam_state(state, state_cov, t, landmarks, map, observed_landmarks, "UKF")

    print("UKF-SLAM finished!")

    print(f"Avg. prediction time: {sum(predict_times)/len(predict_times):.4f} sec.")
    print(f"Avg. correction time: {sum(correct_times)/len(correct_times):.4f} sec.")
    print(f"Avg. step time: {(sum(predict_times) + sum(correct_times))/(len(predict_times) + len(correct_times)):.4f} sec.")

    print(f"Current state vector: {state}")
    # print(f"Current state covariance: {state_cov}")

if __name__ == "__main__":
    main()