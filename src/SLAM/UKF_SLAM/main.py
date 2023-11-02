# Standard
import os
# External
import numpy as np
# Local
from src.utils import read_world, read_data
from src.visualization.plot_slam_state import plot_slam_state
from .prediction import predict
from .correction import correct
from .utils import getLamda
from . import alpha, kappa

def getFx(NUM_LANDMARKS):
    """Calculates the Fx matrix, which is essential to apply the motion model to the pose state, and not the landmarks.
    Fx maps the robot's state from a low dimensional space, to a high dimensional space."""

    Fx = np.zeros((3, 3 + 2*NUM_LANDMARKS))
    Fx[:3, :3] = np.eye(3)

    return Fx

def main():
    # Read world and sensor data
    world_path = os.path.join('exercises', '09_ukf_slam_framework', 'data', "world.dat")
    landmarks = read_world(world_path)
    NUM_LANDMARKS = len(landmarks["ids"])
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
        
    # Initialize observed features
    landmark_history = []
    
    # Get the UKF parameters
    num_dim = state.shape[0]
    gamma = 3

    # Iterate over time
    for i, t in enumerate(data["timesteps"]):
        if i%100 == 0:
            print(f"Iteration: {i}, time: {t}")
            
        # Extract velocity profile from the odometry readings
        dtheta1 = data["odometry"][i][0]
        dr = data["odometry"][i][1]
        dtheta2 = data["odometry"][i][2]
        displacement = np.array([dtheta1, dr, dtheta2]).reshape((3,1))

        # Steps 1-5: Prediction
        state, state_cov = predict(state, state_cov, displacement, process_cov, num_dim, gamma)

        # Get the current landmark observations
        observed_landmarks = data["sensor"][i]

        # Steps 6-23: Correction
        # state, state_cov, landmark_history = correct(expected_state, expected_state_cov, NUM_LANDMARKS, observed_landmarks, landmark_history)

        # Plot robot state
        plot_slam_state(state, state_cov, t, landmarks, landmark_history, observed_landmarks, "UKF")

    print("UKF-SLAM finished!")

    print(f"Current state vector: {state}")
    # print(f"Current state covariance: {state_cov}")

if __name__ == "__main__":
    main()