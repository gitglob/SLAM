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
def predictState(state, u, dt):
    """Predicts the new pose μ of the robot, based on the circular arc velocity model."""

    expected_state = getA(dt)@state + getB(dt)@u

    return expected_state

# Step 3
def predictCovariance(state_covariance, dt, process_cov):
    """Predicts the new covariance matrix."""

    pred_covariance = getA(dt)@state_covariance@getA(dt).T + process_cov

    return pred_covariance

# Prediction step
def predict(state, state_covariance, u, process_cov, dt):
    """Performs the prediction steps of the EKF SLAM algorithm."""

    # Step 2: Predict state
    expected_state = predictState(state, u, dt)

    # Step 3: Expected state covariance
    expected_state_cov = predictCovariance(state_covariance, dt, process_cov)

    return expected_state, expected_state_cov
