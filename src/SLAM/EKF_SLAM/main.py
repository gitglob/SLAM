# Standard
# External
import numpy as np
# Local
from src.utils import read_world, read_data
from src.visualization.plot_slam import plot_state
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
    landmarks = read_world("world.dat")
    NUM_LANDMARKS = len(landmarks["ids"])
    data = read_data("sensor_data.dat")

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
    landmark_history = []

    # Iterate over time
    for i, t in enumerate(data["timesteps"]):
        if i%100 == 0:
            print(f"Iteration: {i}, time: {t}")

        if i==50:
            break
            
        # Extract velocity profile from the odometry readings
        dtheta1 = data["odometry"][i][0]
        dr = data["odometry"][i][1]
        dtheta2 = data["odometry"][i][2]
        displacement = np.array([dtheta1, dr, dtheta2]).reshape((3,1))

        # Steps 1-5: Prediction
        expected_state, expected_state_cov = predict(state, state_cov, displacement, process_cov, Fx, NUM_LANDMARKS)

        # Get the current landmark observations
        observed_landmarks = data["sensor"][i]

        # Steps 6-23: Correction
        state, state_cov, landmark_history = correct(expected_state, expected_state_cov, NUM_LANDMARKS, observed_landmarks, landmark_history)

        # Plot robot state
        plot_state(state, state_cov, t, landmarks, landmark_history, observed_landmarks)

    print("EKF-SLAM finished!")

    print(f"Current state vector: {state}")
    print(f"Current state covariance: {state_cov}")

if __name__ == "__main__":
    main()