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
def predictCovariance(inf_matrix, process_cov, A):
    """Predicts the new covariance matrix."""

    pred_covariance = np.linalg.inv(A @ np.linalg.inv(inf_matrix) @ A.T + process_cov)

    return pred_covariance

# Step 3
def predictState(A, inf_vector, inf_matrix, B, u):
    """Predicts the new pose μ of the robot, based on the circular arc velocity model."""

    expected_state = inf_matrix @ ( A @ np.linalg.inv(inf_matrix) @ inf_vector + B @ u )

    return expected_state

# Prediction step
def predict(inf_matrix, inf_vector, u, process_cov, dt):
    """Performs the prediction steps of the EKF SLAM algorithm."""

    # Calculate A - motion matrix
    A = getA(dt)

    # Calculate B - control matrix
    B = getB(dt)

    # Step 2: Predict state uncertainty
    expected_inf_matrix = predictCovariance(inf_matrix, process_cov, A)

    # Step 3: Predict state
    expected_inf_vector = predictState(A, inf_vector, inf_matrix, B, u)

    return expected_inf_matrix, expected_inf_vector
