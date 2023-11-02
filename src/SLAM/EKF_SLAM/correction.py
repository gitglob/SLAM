# Standard
# External
import numpy as np
# Local
from src.utils import normalize_all_bearings


def getQ(num_landmarks):
    """Returns the uncertainty matrix of the sensors per landmark measured."""
    Q = np.eye(2*num_landmarks) * 0.01

    return Q

def initLandmarkPosition(state, landmark_meas):
    """
    Initialize the estimated position of a new landmark based on the robot's current state and the measurement reading.
    
    This function calculates the position of a landmark by projecting the measured distance and bearing 
    from the robot's current position and orientation.

    Parameters
    ----------
    state : np.ndarray
        The state vector of the robot, typically [x, y, theta], where
        x, y are the coordinates and theta is the orientation.
        
    landmark_meas : np.ndarray
        The measurement vector for the landmark, typically [r, phi], where
        r is the distance to the landmark and phi is the bearing (angle) from the robot's perspective.

    Returns
    -------
    np.ndarray
        The estimated position of the landmark, as a 2x1 vector [x, y].
    """
    # Calculate landmark position based on robot's position and measurement
    lx = state[0] + landmark_meas[0] * np.cos(landmark_meas[1] + state[2])
    ly = state[1] + landmark_meas[0] * np.sin(landmark_meas[1] + state[2])

    # Stack the landmark's x and y position into a single array
    landmark_pos = np.vstack([lx, ly])

    return landmark_pos

def getDelta(state, landmark_id):
    """
    Compute the difference in position between the robot and a specific landmark.

    Parameters
    ----------
    state : np.ndarray
        The state vector containing the robot's pose and the estimated poses of all landmarks.
    landmark_id : int
        The index of the landmark in question.

    Returns
    -------
    np.ndarray
        The delta vector [dx, dy] between the robot's position and the landmark's position.
    """
    delta = state[3 + 2*landmark_id : 3 + 2*landmark_id + 2] - state[:2]
    return delta

def predictObservation(q, delta, state_theta):
    """
    Predict the observation of a landmark from the robot's current state.

    Parameters
    ----------
    q : float
        The squared distance between the robot and the landmark.
    delta : np.ndarray
        The delta vector [dx, dy] between the robot's position and the landmark's position.
    state_theta : float
        The orientation of the robot in the state vector.

    Returns
    -------
    np.ndarray
        The predicted observation vector [range, bearing] of the landmark.
    """
    z_pred_x = np.sqrt(q)
    z_pred_y = np.arctan2(delta[1], delta[0]) - state_theta
    z_pred = np.vstack((z_pred_x, z_pred_y))

    return z_pred

def getFx_landmark(j, NUM_LANDMARKS):
    """
    Construct a matrix that maps the full state space to a state space concerning only one landmark.

    Parameters
    ----------
    j : int
        The landmark index.
    NUM_LANDMARKS : int
        The total number of landmarks in the state vector.

    Returns
    -------
    np.ndarray
        The matrix F that maps the full state to a state with only the j-th landmark.
    """
    F = np.zeros((5, 3+2*NUM_LANDMARKS))
    F[:3, :3] = np.eye(3)
    F[3:5, 3 + 2*j : 3 + 2*j + 2] = np.eye(2)

    return F

def getHi(q, delta, Fx_j):
    """
    Compute the Jacobian matrix of the observation model with respect to the state for a single landmark.

    Parameters
    ----------
    q : float
        The squared distance between the robot and the landmark.
    delta : np.ndarray
        The delta vector [dx, dy] between the robot's position and the landmark's position.
    Fx_j : np.ndarray
        The matrix Fx specific to landmark j that maps the full state to a state concerning only this landmark.

    Returns
    -------
    np.ndarray
        The Jacobian matrix of the observation model H for the specific landmark.
    """
    delta_x = delta[0].item()
    delta_y = delta[1].item()

    H_middle = [[-np.sqrt(q)*delta_x, -np.sqrt(q)*delta_y,  0, np.sqrt(q)*delta_x, np.sqrt(q)*delta_y],
                [delta_y,             -delta_x,            -q, -delta_y,                      delta_x]]
    H_middle = np.array(H_middle).reshape((2, 5))

    H = 1/q * H_middle @ Fx_j

    return H

def getKalmanGain(state_cov, H, Q):
    """
    Compute the Kalman Gain for the state update.

    Parameters
    ----------
    state_cov : np.ndarray
        The state covariance matrix.
    H : np.ndarray
        The Jacobian matrix of the observation model.
    Q : np.ndarray
        The covariance matrix of the observation noise.

    Returns
    -------
    np.ndarray
        The Kalman Gain matrix.
    """

    K = state_cov @ H.T @ np.linalg.inv(H @ state_cov @ H.T + Q)

    return K

def updateExpectedPred(state, K, z, z_pred):
    """
    Update the state prediction with the measurement update step in the Kalman Filter.

    Parameters
    ----------
    state : np.ndarray
        The prior state estimate.
    K : np.ndarray
        The Kalman Gain matrix.
    z : np.ndarray
        The actual observation vector.
    z_pred : np.ndarray
        The predicted observation vector.

    Returns
    -------
    np.ndarray
        The updated state estimate.
    """
    # Calculate and normalize innovation
    innovation = z - z_pred
    innovation_norm = normalize_all_bearings(innovation)

    # Calculate expected state
    state = state + K @ innovation_norm

    return state

def updateExpectedStateCovPred(K, H, state_cov, NUM_LANDMARKS):
    """
    Update the state covariance prediction with the measurement update step in the Kalman Filter.

    Parameters
    ----------
    K : np.ndarray
        The Kalman Gain matrix.
    H : np.ndarray
        The Jacobian matrix of the observation model.
    state_cov : np.ndarray
        The prior state covariance matrix.
    NUM_LANDMARKS : int
        The total number of landmarks in the state vector.

    Returns
    -------
    np.ndarray
        The updated state covariance matrix.
    """
    I = np.eye(NUM_LANDMARKS*2+3, NUM_LANDMARKS*2+3)

    state_cov = (I - K @ H) @ state_cov

    return state_cov

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
            expected_state[3 + 2*j : 3 + 2*j + 2] = initLandmarkPosition(expected_state, z)
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
    expected_state_cov = updateExpectedStateCovPred(K, H, expected_state_cov, NUM_LANDMARKS)

    # Step 22: Return the new state and state covariance estimation
    return expected_state, expected_state_cov, landmark_history
