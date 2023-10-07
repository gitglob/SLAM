# External
import numpy as np
# Local
from . import NUM_LANDMARKS


def get_Q_t(sigma_r=100, sigma_phi=100):
    """Returns the uncertainty matrix of the sensors"""
    Q_t = [[sigma_r**2, 0], 
           [0,          sigma_phi**2]]
    Q_t = np.array(Q_t).reshape(2,2)

    return Q_t

def initialize_landmark_position(mi, r, phi):
    """Initializes the position of a landmark if it has not been seen before."""
    mi_j_x = mi[0] + r*np.cos(phi + mi[2])
    mi_j_y = mi[1] + r*np.sin(phi + mi[2])

    mi = np.array([mi_j_x, mi_j_y]).reshape(2,1)

    return mi

def get_delta(expected_state, expected_state_landmark):
    return expected_state[:2] - expected_state_landmark

def predict_observation(q, delta, mi_theta):
    z_pred_x = np.sqrt(q)
    z_pred_y = np.arctan2(delta[0], delta[1]) - mi_theta
    z_pred = np.array([z_pred_x, z_pred_y]).reshape((2,1))

    return z_pred

# Creates and returns the F matrix
def getF(j):
    F_upper_left = np.eye(3)
    F_upper_right = np.zeros((3, 2*j -2 +2 +2*NUM_LANDMARKS - 2*j))
    F_upper = np.hstack((F_upper_left, F_upper_right))
    F_lower_left = np.zeros((2, 3 + 2*j -2))
    F_lower_center = np.eye(2)
    F_lower_right = np.zeros((2*NUM_LANDMARKS - 2*j))
    F_lower = np.hstack((F_lower_left, F_lower_center, F_lower_right))
    F = np.vstack(F_upper, F_lower)

    return F

def getH(q, delta, F):
    H_middle = [[-np.sqrt(q)*delta[0], -np.sqrt(q)*delta[1],  0, np.sqrt(q)*delta[0], np.sqrt(q)*delta[1]],
                [delta[1],             -delta[0],            -q, -delta[1],           delta[0]]]
    H_middle = np.array(H_middle).reshape((2, 5))
    H = 1/q @ H_middle @ F

    return H

def getKalmanGain(expected_state_cov, H, Q):
    """Calculates the Kalman Gain."""
    K = expected_state_cov @ H.T @ np.linalg.inv(H @ expected_state_cov @ H.T + Q)

    return K

def updateStateCovPred(K, H, expected_state_cov):
    """Updates the state uncertainty using the Kalman Gain and the Jacobian of the function that computes the expected observation."""
    I = np.eye(NUM_LANDMARKS*2+3, NUM_LANDMARKS*2+3)

    expected_state_cov = (I - K @ H) @ expected_state_cov

    return expected_state_cov

def update_state_pred(state_pred, K, z, z_pred):
    """Updates the state prediction using the previous state prediction, the Kalman Gain and the real and expected observation of a specific landmark."""

    state_pred = state_pred + K @ (z - z_pred)

    return state_pred


def correct(expected_state, expected_state_cov):
    """Performs the correction steps of the EKF SLAM algorithm."""
    # Step 6
    Q = get_Q_t()

    # List of observed landmarks
    j_seen = []

    # Step 2
    observed_features = [[0,0], [1,1], [2,2], [3,3], [4,4]] # observed features
    # Step 7
    for i, z in enumerate(observed_features):
        r, phi = z

        # Step 8
        j = i
        # Step 9
        if j not in j_seen:
            # Step 10
            mi_landmark_pred = initialize_landmark_position(expected_state, r, phi)
        
        # Step 11: endif

        # Step 12
        delta = get_delta(expected_state, mi_landmark_pred)

        # Step 13
        q = delta.T@delta

        # Step 14: Expected Observation
        z_pred = predict_observation(q, delta, expected_state[2])

        # Step 15
        F = getF(j)

        # Step 16: Jacobian of Expected Observation
        H = getH(q, delta, F)

        # Step 17: Kalman Fain
        K = getKalmanGain(expected_state_cov, H, Q)

        # Step 18: Expected State
        state_pred = update_state_pred(state_pred, K, z, z_pred)

        # Step 19: Expected Uncertainty Update
        expected_state_cov = updateStateCovPred(K, H, expected_state_cov)

    # Step 20: endfor

    # Step 20: Updated state
    state = state_pred

    # Step 21: Updated state covariance
    state_cov = expected_state_cov

    return state, state_cov
