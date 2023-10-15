# Standard
# External
import numpy as np
# Local

def getA(dt):
    """Calculates the A matrix, which is the linear motion model."""

    A = [[1, 0, dt,  0],
         [0, 1,  0, dt],
         [0, 0,  1,  0],
         [0, 0,  0,  1]]
    A = np.array(A).reshape(4,4)

    return A

def getB(dt):
    """Return the B matrix, which is the control matrix."""
    B = [[0.5*dt**2,         0],
         [0        , 0.5*dt**2],
         [dt       ,         0],
         [0        ,        dt]]
    B = np.array(B).reshape((4,2))

    return B

# Step 2
def predictState(A, state, B, u):
    """Predicts the new pose μ of the robot, based on the circular arc velocity model."""

    expected_state = A@state + B@u

    return expected_state

# Step 3
def predictCovariance(state_covariance, process_cov, A, dt):
    """Predicts the new covariance matrix."""

    pred_covariance = A@state_covariance@A.T + process_cov

    return pred_covariance

# Prediction step
def predict(state, state_covariance, u, process_cov, dt):
    """Performs the prediction steps of the EKF SLAM algorithm."""

    # Calculate A - motion matrix
    A = getA(dt)

    # Calculate B - control matrix
    B = getB(dt)

    # Step 2: Predict state
    expected_state = predictState(A, state, B, u)

    # Step 3: Expected state covariance
    expected_state_cov = predictCovariance(state_covariance, process_cov, A, dt)

    return expected_state, expected_state_cov
