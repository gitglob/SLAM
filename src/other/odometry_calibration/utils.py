# Standard
# External
import numpy as np
# Local


def v2t(v):
    """
    Compute the homogeneous transformation matrix 'A' from the pose vector 'v'.

    Parameters
    ----------
    v : np.ndarray
        The pose vector in the format [x, y, theta] where 'x' and 'y' are 2D position coordinates,
        and 'theta' is the orientation angle in radians.

    Returns
    -------
    np.ndarray
        The homogeneous transformation matrix corresponding to the pose vector.
    """
    tx = v[0].item()
    ty = v[1].item()
    c = np.cos(v[2]).item()
    s = np.sin(v[2]).item()
    A = np.array([[c, -s, tx],
                  [s,  c, ty],
                  [0,  0,  1]])

    return A

def t2v(A):
    """
    Compute the pose vector 'v' from the homogeneous transformation matrix 'A'.

    Parameters
    ----------
    A : np.ndarray
        The homogeneous transformation matrix.

    Returns
    -------
    np.ndarray
        The pose vector in the format [x, y, theta], where 'x' and 'y' are the 2D position coordinates,
        and 'theta' is the orientation angle in radians.
    """
    v = np.empty((3, 1))
    v[0:2, 0] = A[0:2, 2]
    v[2, 0] = np.arctan2(A[1, 0], A[0, 0])
    return v

def compute_trajectory(U):
    """
    Computes the trajectory of the robot by chaining up the incremental movements 
    of the odometry vector.

    Parameters
    ----------
    U : numpy.ndarray
        A Nx3 matrix, where each row contains the odometry ux, uy, utheta.

    Returns
    -------
    numpy.ndarray
        A (N+1)x3 matrix, where each row contains the robot position 
        (starting from [0, 0, 0]).
    """
    # Initialize the trajectory matrix
    T = np.zeros((U.shape[0]+1, 3, 1))
    # Store the first pose in the result
    T[0, :] = np.zeros((3,1))
    # The current pose in the chain
    currentPose = v2t(T[0, :])

    # Compute the result of chaining up the odometry deltas
    for i in range(U.shape[0]):
        # Compute the current pose of the robot at i
        currentPose = currentPose @ v2t(U[i, :])
        # Add this value to the T matrix
        T[i + 1, :] = t2v(currentPose)

    return T

def apply_odometry_correction(X, U):
    """
    Computes a calibrated vector of odometry measurements by applying the 
    bias term (X) to each line of the measurements.

    Parameters
    ----------
    X : numpy.ndarray
        A 3x3 matrix obtained by the calibration process.
    U : numpy.ndarray
        A Nx3 matrix containing the odometry measurements.

    Returns
    -------
    numpy.ndarray
        A Nx3 matrix containing the corrected odometry measurements.
    """
    C = (X @ U.T).T
    return C

def ls_calibrate_odometry(Z):
    """
    Solves the odometry calibration problem given a measurement matrix Z. 
    The function assumes that the information matrix is the identity for each of the measurements.

    Parameters
    ----------
    Z : numpy.ndarray
        The measurement matrix, where each row contains [u'x, u'y, u'theta, ux, uy, utheta].

    Returns
    -------
    numpy.ndarray
        The calibration matrix X, which corrects odometry measurements.
    """
    # initial solution (the identity transformation)
    X = np.eye(3)

    # Initialize H and b of the linear system
    H = np.zeros((9, 9))
    b = np.zeros(9)

    # Initialize info matrix, omega
    omega = np.eye(3)

    # Loop through the measurements and update H and b
    for i in range(Z.shape[0]):
        # Compute jacobian and error for this point
        J = jacobian(i, Z)
        e = error_function(i, X, Z)
        
        # Update H and b
        H += J.T @ omega @ J
        b += J.T @ omega.T @ e

    # Solve and update the solution
    delta_x = - np.linalg.inv(H) @ b
    X += delta_x.reshape(3, 3)

    return X

def error_function(i, X, Z):
    """
    Computes the error of the i-th measurement in Z given the calibration parameters.

    Parameters
    ----------
    i : int
        The index of the measurement in Z.
    X : numpy.ndarray
        The current calibration parameters, a 3x3 matrix.
    Z : numpy.ndarray
        The measurement matrix, where each row contains first the scan-match result
        and then the motion reported by odometry.

    Returns
    -------
    numpy.ndarray
        The error of the i-th measurement, a 3-element vector.
    """

    # Compute the error of each measurement
    u_truth = Z[i, 0:3]
    u_odom = Z[i, 3:6]
    e = u_truth - X @ u_odom
    return e

def jacobian(i, Z):
    """
    Computes the Jacobian of the error function for the i-th measurement in Z.

    Parameters
    ----------
    i : int
        The index of the measurement in Z.
    Z : numpy.ndarray
        The measurement matrix.

    Returns
    -------
    numpy.ndarray
        The Jacobian of the i-th measurement, a 3x9 matrix.
    """

    # Compute the Jacobian
    J = np.zeros((3, 9))
    u_values = Z[i, 3:6]
    J[0, 0:3] = u_values
    J[1, 3:6] = u_values
    J[2, 6:9] = u_values
    J = -J
    return J

