# Standard
# External
import numpy as np
# Local
from src.utils import normalize_all_bearings


def getQ(num_landmarks):
    """Returns the uncertainty matrix of the sensors per landmark measured."""
    Q = np.eye(2*num_landmarks) * 0.01

    return Q

def initializeLandmarkPosition(expected_robot_state, landmark_reading):
    """Initializes the position of a landmark if it has not been seen before."""
    expected_state_landmark_x = expected_robot_state[0] + landmark_reading[0]*np.cos(landmark_reading[1] + expected_robot_state[2])
    expected_state_landmark_y = expected_robot_state[1] + landmark_reading[0]*np.sin(landmark_reading[1] + expected_robot_state[2])

    expected_state_landmark = np.vstack([expected_state_landmark_x, expected_state_landmark_y])

    return expected_state_landmark

def getDelta(expected_state, landmark_id):
    """Returns the difference between expected robot's position and the landmark's expected position."""
    delta = expected_state[3 + 2*landmark_id : 3 + 2*landmark_id + 2] - expected_state[:2]
    return delta

def predictObservation(q, delta, state_theta):
    """Returns the expected observation based on the expected robot's pose and the expected specific landmark's pose."""
    z_pred_x = np.sqrt(q)
    z_pred_y = np.arctan2(delta[1], delta[0]) - state_theta
    z_pred = np.vstack((z_pred_x, z_pred_y))

    return z_pred

def getFx_landmark(j, NUM_LANDMARKS):
    "Returns the F matrix, which maps the world state (robot + landmarks) to the robot state (only robot)."
    F = np.zeros((5, 3+2*NUM_LANDMARKS))
    F[:3, :3] = np.eye(3)
    F[3:5, 3 + 2*j - 2 : 3 + 2*j] = np.eye(2)

    return F

def getHi(q, delta, Fx_j):
    """Returns the Jacobian of the expected observation function, for the specific landmark and the current robot's pose."""
    delta_x = delta[0].item()
    delta_y = delta[1].item()

    H_middle = [[-np.sqrt(q)*delta_x, -np.sqrt(q)*delta_y,  0, np.sqrt(q)*delta_x, np.sqrt(q)*delta_y],
                [delta_y,             -delta_x,            -q, -delta_y,                      delta_x]]
    H_middle = np.array(H_middle).reshape((2, 5))

    H = 1/q * H_middle @ Fx_j

    return H

def getKalmanGain(expected_state_cov, H, Q):
    """Calculates the Kalman Gain."""
    K = expected_state_cov @ H.T @ np.linalg.inv(H @ expected_state_cov @ H.T + Q)

    return K

def updateExpectedPred(expected_state, K, z, z_pred):
    """Updates the state prediction using the previous state prediction, the Kalman Gain and the real and expected observation of a specific landmark."""

    # Calculate and normalize innovation
    innovation = z - z_pred
    innovation_norm = normalize_all_bearings(innovation)

    # Calculate expected state
    expected_state = expected_state + K @ innovation_norm

    return expected_state

def updateStateCovPred(K, H, expected_state_cov, NUM_LANDMARKS):
    """Updates the state uncertainty using the Kalman Gain and the Jacobian of the function that computes the expected observation."""
    I = np.eye(NUM_LANDMARKS*2+3, NUM_LANDMARKS*2+3)

    expected_state_cov = (I - K @ H) @ expected_state_cov

    return expected_state_cov

# Step B: Correction
def correct(expected_state, expected_state_cov, NUM_LANDMARKS, observed_landmarks, landmark_history):
    """Performs the correction steps of the EKF SLAM algorithm."""
    # Step 6: Get the number of observations
    num_observed_landmarks = len(observed_landmarks)

    # Step 7: Initialize landmark real and expected expected observations
    zs = np.zeros((2*num_observed_landmarks, 1))
    expected_zs = np.zeros((2*num_observed_landmarks, 1))

    # Step 7: Initialize empty Jacobian
    H = np.zeros((2*num_observed_landmarks, 3 + 2*NUM_LANDMARKS))

    # Step 8: Initialize landmark measurement covariance
    Q = getQ(num_observed_landmarks)

    # Step 9: Iterate over landmark observations
    for i, feature in enumerate(observed_landmarks):
        # Step 10a: Extract landmark id and position
        j = feature[0]
        z = np.vstack(feature[1:3])

        # Step 10b: Append landmark observed position to the array of total landmark observations
        zs[2*i : 2*(i+1)] = z

        # Step 10: Check for new observations
        if j not in landmark_history:
            print(f"New landmark: {j}")
            # Step 11: Initialize landmark predictions if they are first-seen now
            expected_state[3 + 2*j : 3 + 2*j + 2] = initializeLandmarkPosition(expected_state, z)
            landmark_history.append(j)
        
            # Step 12: endif

        # Step 13
        delta = getDelta(expected_state, j)

        # Step 14
        q = (delta.T@delta).item()

        # Step 15a: Expected Observation
        expected_z = predictObservation(q, delta, expected_state[2])

        # Step 15b: Append the expected observation of the current landmark to the total observation array
        expected_zs[2*i : 2*(i+1)] = expected_z

        # Step 16
        Fx_j = getFx_landmark(j, NUM_LANDMARKS)

        # Step 17a: Jacobian of Expected Observation
        Hi = getHi(q, delta, Fx_j)

        # Step 17b: Append the jacobian of the current landmark to the total jacobian matrix
        H[2*i : 2*(i+1), :] = Hi

    # Step 18: endfor

    # Step 19: Kalman Fain
    K = getKalmanGain(expected_state_cov, H, Q)

    # Step 20: Expected State
    expected_state = updateExpectedPred(expected_state, K, zs, expected_zs)

    # Step 21: Expected Uncertainty Update
    expected_state_cov = updateStateCovPred(K, H, expected_state_cov, NUM_LANDMARKS)

    # Step 22: Return the new state and state covariance estimation
    return expected_state, expected_state_cov, landmark_history
