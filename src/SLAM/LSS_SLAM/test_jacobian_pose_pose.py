# Standard
# Extermal
import numpy as np
# Local
from .functions import linearize_pose_landmark_constraint


def numerical_jacobian(f, x, delta=1e-6):
    """
    Computes the numerical Jacobian of a function at a given point.

    Parameters
    ----------
    f : function
        The function for which to compute the Jacobian. It must take a single numpy array as input.
    x : numpy.ndarray
        The point at which to compute the Jacobian.
    delta : float, optional
        The finite difference delta, by default 1e-6.

    Returns
    -------
    numpy.ndarray
        The numerical Jacobian matrix.
    """
    n = len(x)
    jac = np.zeros((2, n))
    for i in range(n):
        xp = np.copy(x)
        xm = np.copy(x)
        xp[i] += delta
        xm[i] -= delta
        jac[:, i] = (f(xp) - f(xm)) / (2 * delta)
    return jac

def main():
    epsilon = 1e-5

    x1 = np.array([1.1, 0.9, 1])
    x2 = np.array([2.2, 1.9])
    z = np.array([1.3, -0.4])

    # Get the analytic Jacobian
    e, A, B = linearize_pose_landmark_constraint(x1, x2, z)

    # Check the error vector
    e_true = np.array([0.135804, 0.014684])
    if np.linalg.norm(e - e_true) > epsilon:
        print('Your error function seems to return a wrong value')
        print('Result of your function:', e)
        print('True value:', e_true)
    else:
        print('The computation of the error vector appears to be correct')

    # Test for x1
    f_x1 = lambda x: linearize_pose_landmark_constraint(x, x2, z)[0]
    ANumeric = numerical_jacobian(f_x1, x1)

    diff = ANumeric - A
    if np.max(np.abs(diff)) > epsilon:
        print('Error in the Jacobian for x1')
        print('Your analytic Jacobian:', A)
        print('Numerically computed Jacobian:', ANumeric)
        print('Difference:', diff)
    else:
        print('Jacobian for x1 appears to be correct')

    # Test for x2
    f_x2 = lambda x: linearize_pose_landmark_constraint(x1, x, z)[0]
    BNumeric = numerical_jacobian(f_x2, x2, delta=1e-6)

    diff = BNumeric - B
    if np.max(np.abs(diff)) > epsilon:
        print('Error in the Jacobian for x2')
        print('Your analytic Jacobian:', B)
        print('Numerically computed Jacobian:', BNumeric)
        print('Difference:', diff)
    else:
        print('Jacobian for x2 appears to be correct')

if __name__ == "__main__":
    main()
    