# Standard
# External
import numpy as np
# Local
from src.utils import normalize_angle
from .utils import getSigmaPoints, getWeight


def initLandmarkPosition(state, state_cov, z, Q, gamma):
    """
    Initializes a landmark based on its first observation.
    
    Parameters:
    state (np.ndarray): The mean state vector.
    state_cov (np.ndarray): The state covariance matrix.
    z (dict): The observation containing 'range', 'bearing', and 'id'.
    Q (np.ndarray): The sensor noise covariance matrix.
    gamma (float): The scaling factor for computing state_cov points.
    
    Returns:
    tuple: Tuple containing the updated mean vector, covariance matrix, and map of landmarks.
    """
    
    # Append the landmark measurement to the state vector
    state = np.vstack((state, z))
    
    # Initialize the landmark measurement uncertainty and add it to state_cov
    combined_size = state_cov.shape[0] + Q.shape[0]
    new_state_cov = np.zeros((combined_size, combined_size))
    new_state_cov[:state_cov.shape[0], :state_cov.shape[1]] = state_cov # Covariance for existing state
    new_state_cov[state_cov.shape[0]:, state_cov.shape[1]:] = Q # Covariance for new landmark
    state_cov = new_state_cov
    
    # Compute sigma points
    sigma_points = getSigmaPoints(state, state_cov, gamma)
    # Normalize angles in sigma points
    for i in range(len(sigma_points)):
        sigma_points[i,2,0] = normalize_angle(sigma_points[i, 2, 0])
    
    # For each sigma point, compute the x/y location of the new landmark, based on the robot's location
    newX = np.zeros((len(sigma_points),1))
    newY = np.zeros((len(sigma_points),1)) 
    for i in range(len(sigma_points)):
        newX[i] = sigma_points[i,0] + sigma_points[i,-2] * np.cos(sigma_points[i,2] + sigma_points[i,-1])
        newY[i] = sigma_points[i,1] + sigma_points[i,-2] * np.sin(sigma_points[i,2] + sigma_points[i,-1])
    
    # For each sigma point, replace the last 2 components of the sigma points with the x/y position of the landmark
    sigma_points[:,-2] = newX
    sigma_points[:,-1] = newY
    
    # Compute the new state dimension and lamda
    num_dim = len(state)
    lamda = gamma - num_dim
    
    # Recover the updated mean vector state
    state = np.zeros((num_dim, 1))
    for i in range(len(sigma_points)):
        w_m = getWeight(lamda, gamma, i)
        state += sigma_points[i] * w_m
    
    # Recover angle by summing up the sines and cosines
    cosines = 0
    sines = 0
    for i in range(len(sigma_points)):
        w_m = getWeight(lamda, gamma, i)
        cosines += np.cos(sigma_points[i,2]) * w_m
        sines += np.sin(sigma_points[i,2]) * w_m
    state[2] = normalize_angle(np.arctan2(sines, cosines))
    
    # Recover the updated state covariance
    state_cov = np.zeros((num_dim, num_dim))
    for i in range(len(sigma_points)):
        w_c = getWeight(lamda, gamma, i)
        sigma_points_diff = sigma_points[i] - state
        sigma_points_diff[2] = normalize_angle(sigma_points_diff[2])
        state_cov += w_c * sigma_points_diff @ sigma_points_diff.T

    return state, state_cov

def predictObservation(sigma_points, l_idx):
    """
    Compute the expected observation of the specific landmark for all sigma points.

    Essentially passes the sigma points through the non-linear measurement function
    that converts polar coordinates to cartesian.
    
    Parameters:
    ----------
    sigma_points : np.ndarray 
        The matrix of sigma points.
    l_idx : int 
        The index of the landmark.
    
    Returns:
    -------
    np.ndarray
        The matrix of predicted measurement for every sigma point.
    """
    propagated_sigma_points = np.zeros((sigma_points.shape[0], 2, 1))

    # Get the the sigma points for the specific landmark
    landmarkXs = sigma_points[:, 2*l_idx + 2]
    landmarkYs = sigma_points[:, 2*l_idx + 3]

    for i in range(len(propagated_sigma_points)):
        # Get range and bearing by computing the differences between the landmark positions and robot position
        r = np.sqrt((landmarkXs[i] - sigma_points[i, 0])**2 + (landmarkYs[i] - sigma_points[i, 1])**2)
        bearing = np.arctan2(landmarkYs[i] - sigma_points[i, 1], landmarkXs[i] - sigma_points[i, 0]) - sigma_points[i, 2]
        bearing = normalize_angle(bearing)

        propagated_sigma_points[i] = np.vstack([r, bearing])
    
    return propagated_sigma_points

# Step 8
def recoverObservation(propagated_sigma_points, lamda, gamma):
    """
    Recover the weighted mean of the propagated sigma points to
    calculate the mean expected observation.
    
    Parameters
    ----------
    propagated_sigma_points : np.ndarray
        Sigma points after being passed through the motion model.
    lamda : float
        Scaling parameter for the sigma points and weights.
    num_dim : int
        Dimension of the state vector.

    Returns
    -------
    np.ndarray
        Predicted state of the system.
    """

    expected_observation = np.zeros((2,1))
    expected_x = 0
    expected_y = 0
    for i in range(len(propagated_sigma_points)):
        w_m = getWeight(lamda, gamma, i)
        expected_observation += w_m * propagated_sigma_points[i]
        expected_x += w_m * np.cos(propagated_sigma_points[i,1])
        expected_y += w_m * np.sin(propagated_sigma_points[i,1])

    expected_observation[1] = normalize_angle(np.arctan2(expected_y, expected_x))

    return expected_observation

# Step 9
def updateUncertainty(propagated_sigma_points, expected_observation, measurement_cov, lamda, gamma, num_dim):
    """Calculates S."""
    S = np.zeros(measurement_cov.shape)
    for i in range(len(propagated_sigma_points)):
        w_c = getWeight(lamda, gamma, i)
        diff = propagated_sigma_points[i] - expected_observation
        diff[1] = normalize_angle(diff[1])
        S += w_c * diff @ diff.T + measurement_cov

    return S

# Step 10
def predictStateCov(sigma_points, expected_state, propagated_sigma_points, expected_observation, lamda, gamma, num_dim):
    """Calculates the expected state uncertainty."""
    expected_state_cov = np.zeros((num_dim, 2))
    for i in range(len(sigma_points)):
        w_c = getWeight(lamda, gamma, i)

        diff_state = sigma_points[i] - expected_state
        diff_state[2] = normalize_angle(diff_state[2])

        diff_observation = propagated_sigma_points[i] - expected_observation
        diff_observation[1] = normalize_angle(diff_observation[1])
        
        expected_state_cov += w_c * diff_state @ diff_observation.T

    return expected_state_cov

# Step 11
def getKalmanGain(expected_state_cov, S):
    """Calculates the Kalman Gain."""
    K = expected_state_cov @ np.linalg.inv(S)

    return K

# Step 12
def updateState(expected_state, K, observation, expected_observation):
    """Updates the robot's state."""
    # Calculate innovation
    innovation = observation - expected_observation
    innovation[1] = normalize_angle(innovation[1])

    # Update state
    state = expected_state + K @ innovation
    state[2] = normalize_angle(state[2])

    return state

# step 13
def updateStateCov(expected_state_cov, K, S):
    """Updates the state covariance matrrix."""
    state_cov = expected_state_cov - K @ S @ K.T

    return state_cov


# Step B: Correction
def correct(expected_state, expected_state_cov, measurement_cov, observed_landmarks, map, gamma):
    """Performs the correction steps of the UKF SLAM algorithm."""

    # Step 6: Iterate over landmark observations
    for landmark in observed_landmarks:
        # Step 7: Extract landmark id and position
        j = landmark[0]
        z = np.vstack(landmark[1:3])

        # Step 8: Check for new observations
        if j not in map:
            print(f"New landmark: {j}")
            # Step 9: Add landmark to map if it is first seen now
            expected_state, expected_state_cov = initLandmarkPosition(expected_state, expected_state_cov, z, measurement_cov, gamma)
            map.append(j)
        
            # Step 10: endif

        # Step 11: Compute sigma points
        sigma_points = getSigmaPoints(expected_state, expected_state_cov, gamma)
        for k in range(len(sigma_points)):
            sigma_points[k,2,0] = normalize_angle(sigma_points[k,2,0]) # Normalize angle

        # Step 12: Update dimension size and lamda
        num_dim = len(expected_state)
        lamda = gamma - num_dim

        # Step 13: State update
        landmark_idx = map.index(j)
        propagated_sigma_points = predictObservation(sigma_points, landmark_idx)

        # Step 14: Observation prediction
        expected_observation = recoverObservation(propagated_sigma_points, lamda, gamma)

        # Step 15: S
        S = updateUncertainty(propagated_sigma_points, expected_observation, measurement_cov, lamda, gamma, num_dim)

        # Step 16: Uncertainty Update
        expected_cross_cov = predictStateCov(sigma_points, expected_state, propagated_sigma_points, expected_observation, lamda, gamma, num_dim)

        # Step 17: Kalman Fain
        K = getKalmanGain(expected_cross_cov, S)

        # Step 28: Expected State
        expected_state = updateState(expected_state, K, z, expected_observation)

        # Step 19: Expected Uncertainty Update
        expected_state_cov = updateStateCov(expected_state_cov, K, S)
        
        # Step 20: endfor

    # Step 21: Return the new state and state covariance estimation
    return expected_state, expected_state_cov, map
