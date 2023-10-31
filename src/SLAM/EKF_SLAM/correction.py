# Standard
# External
import numpy as np
# Local


def getQ(sigma_r=100, sigma_phi=100):
    """Returns the uncertainty matrix of the sensors"""
    Q_t = [[sigma_r**2, 0], 
           [0,          sigma_phi**2]]
    Q_t = np.array(Q_t).reshape(2,2)

    return Q_t

def initializeLandmarkPosition(expected_robot_state, landmark_state):
    """Initializes the position of a landmark if it has not been seen before."""
    expected_state_landmark_x = expected_robot_state[0] + landmark_state[0]*np.cos(landmark_state[1] + expected_robot_state[2])
    expected_state_landmark_y = expected_robot_state[1] + landmark_state[0]*np.sin(landmark_state[1] + expected_robot_state[2])

    expected_state_landmark = np.array([expected_state_landmark_x, expected_state_landmark_y]).reshape(2,1)

    return expected_state_landmark

def getDelta(expected_state, expected_state_landmark):
    """Returns the difference between expected robot's position and the expected landmark's position."""
    delta = (expected_state[:2] - expected_state_landmark).reshape(2, 1)
    return delta

def predictObservation(q, delta, state_theta):
    """Returns the expected observation based on the expected robot's pose and the expected specific landmark's pose."""
    z_pred_x = np.sqrt(q)
    z_pred_y = np.arctan2(delta[0], delta[1]) - state_theta
    z_pred = np.vstack((z_pred_x, z_pred_y))

    return z_pred

def getFx_landmark(j, NUM_LANDMARKS):
    "Returns the F matrix, which maps the world state (robot + landmarks) to the robot state (only robot)."
    F_upper_left = np.eye(3)
    F_upper_right = np.zeros((3, 2*j-2 +2 +2*(NUM_LANDMARKS - j)))
    F_upper = np.hstack((F_upper_left, F_upper_right))

    F_lower_left = np.zeros((2, 3 + 2*j -2))
    F_lower_center = np.eye(2)
    F_lower_right = np.zeros((2, 2*(NUM_LANDMARKS - j)))
    F_lower = np.hstack((F_lower_left, F_lower_center, F_lower_right))
    
    F = np.vstack((F_upper, F_lower))

    return F

def getH(q, delta, F):
    """Returns the Jacobian of the expected observation function, for the specific landmark and the current robot's pose."""
    delta_x = delta[0].item()
    delta_y = delta[1].item()

    H_middle = [[-np.sqrt(q)*delta_x, -np.sqrt(q)*delta_y,  0, np.sqrt(q)*delta_x, np.sqrt(q)*delta_y],
                [delta_y,             -delta_x,            -q, -delta_y,           delta_x]]
    H_middle = np.array(H_middle).reshape((2, 5))

    H = 1/q * H_middle @ F

    return H

def getKalmanGain(expected_state_cov, H, Q):
    """Calculates the Kalman Gain."""
    K = expected_state_cov @ H.T @ np.linalg.inv(H @ expected_state_cov @ H.T + Q)

    return K

def updateStateCovPred(K, H, expected_state_cov, NUM_LANDMARKS):
    """Updates the state uncertainty using the Kalman Gain and the Jacobian of the function that computes the expected observation."""
    I = np.eye(NUM_LANDMARKS*2+3, NUM_LANDMARKS*2+3)

    expected_state_cov = (I - K @ H) @ expected_state_cov

    return expected_state_cov

def updateExpectedPred(expected_state, K, z, z_pred):
    """Updates the state prediction using the previous state prediction, the Kalman Gain and the real and expected observation of a specific landmark."""

    expected_state = expected_state + K @ (z - z_pred)

    return expected_state


def correct(expected_state, expected_state_cov, NUM_LANDMARKS, observed_features, all_seen_features):
    """Performs the correction steps of the EKF SLAM algorithm."""
    # Step 6
    Q = getQ()

    # Step 7
    for feature in observed_features:
        z = np.vstack(feature[1:3])

        # Step 8
        j = feature[0]

        # Step 9
        if j not in all_seen_features:
            print(f"New landmark: {j}")

            # Step 10
            mi_landmark_pred = initializeLandmarkPosition(expected_state, z)
            all_seen_features.append(j)
        
            # Step 11: endif
        else:
            mi_landmark_pred = expected_state[3 + 2*j : 3 + 2*j + 2]

        # Step 12
        delta = getDelta(expected_state, mi_landmark_pred)

        # Step 13
        q = (delta.T@delta).item()

        # Step 14: Expected Observation
        z_pred = predictObservation(q, delta, expected_state[2])

        # Step 15
        Fx_landmark = getFx_landmark(j, NUM_LANDMARKS)

        # Step 16: Jacobian of Expected Observation
        H = getH(q, delta, Fx_landmark)

        # Step 17: Kalman Fain
        K = getKalmanGain(expected_state_cov, H, Q)

        # Step 18: Expected State
        expected_state = updateExpectedPred(expected_state, K, z, z_pred)

        # Step 19: Expected Uncertainty Update
        expected_state_cov = updateStateCovPred(K, H, expected_state_cov, NUM_LANDMARKS)

    # Step 20: endfor

    # Step 20: Updated state
    state = expected_state

    # Step 21: Updated state covariance
    state_cov = expected_state_cov

    return state, state_cov
