# External
import numpy as np
# Local
from src.utils import read_world, read_data
from src.visualization.plot_slam import plot_state
from .prediction import predict
from .correction import correct


def main():
    # Read world and sensor data
    landmarks = read_world("world.dat")
    NUM_LANDMARKS = len(landmarks["ids"])
    data = read_data("sensor_data.dat")

    # Initialize state
    state = np.zeros((3 + NUM_LANDMARKS*2, 1))

    # Initialize state covariance
    state_cov = np.zeros((3 + NUM_LANDMARKS*2, 3 + NUM_LANDMARKS*2))
    state_cov[3:3 + NUM_LANDMARKS*2, 3:3 + NUM_LANDMARKS*2] = np.eye(NUM_LANDMARKS*2) * 1000
    
    # Initialize process noise
    process_cov = np.zeros((3 + NUM_LANDMARKS*2, 3 + NUM_LANDMARKS*2))
    process_cov[0,0] = 0.1
    process_cov[1,1] = 0.1
    process_cov[2,2] = 0.01
    
    # Initialize observed features
    all_seen_features = []

    # Iterate over time
    for i, t in enumerate(data["timesteps"]):
        if i%100 == 0:
            print(f"Iteration: {i}, time: {t}")
            
        # Calculate dt
        if i == 0:
            continue
        else:
            dt = t - data["timesteps"][i-1]

        # Generate random control input (linear/rotational velocity)
        omega1 = (data["odometry"][i][0] - data["odometry"][i-1][0]) / dt
        v = (data["odometry"][i][1] - data["odometry"][i-1][1]) / dt
        omega2 = (data["odometry"][i][2] - data["odometry"][i-1][2]) / dt
        u = np.array([omega1, v, omega2]).reshape((3,1))

        # Steps 1-5: Prediction
        expected_state, expected_state_cov = predict(state, state_cov, u, process_cov, dt, NUM_LANDMARKS)
        
        # Steps 6-23: Correction
        state, state_cov = correct(expected_state, expected_state_cov, NUM_LANDMARKS, data["sensor"][i], all_seen_features)

        # Plot robot state
        plot_state(state, state_cov, landmarks, t, all_seen_features, data["sensor"][i])

    print("EKF-SLAM finished!")

    print(f"Current state vector: {state}")
    # print(f"Current state covariance: {state_cov}")

if __name__ == "__main__":
    main()