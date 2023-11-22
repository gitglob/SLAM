# Standard
# External
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.linalg import sqrtm
# Local


def drawellipse(x, a, b, color):
    """
    Draw an ellipse with center x, half-axes a and b, inclination angle theta, and specified color.
    
    Parameters:
    x : array_like
        The [x, y, theta] ellipse center coordinates and rotation angle in radians.
    a : float
        The length of the semi-major axis.
    b : float
        The length of the semi-minor axis.
    color : str or tuple
        The color specifier, either as a string or as an RGB tuple.
        
    Returns:
    h : matplotlib.lines.Line2D
        The handle to the line object representing the ellipse on the plot.
    """
    
    # Constants
    NPOINTS = 100  # Point density or resolution
    
    # Compose point vector
    ivec = np.linspace(0, 2 * np.pi, NPOINTS)  # Index vector
    p = np.array([a * np.cos(ivec),  # 2 x n matrix which
                  b * np.sin(ivec)])  # holds ellipse points
    
    # Translate and rotate
    xo, yo, angle = x
    R = np.array([[np.cos(angle), -np.sin(angle)],  # Rotation matrix
                  [np.sin(angle),  np.cos(angle)]])
    T = np.array([[xo], [yo]]) @ np.ones((1, NPOINTS))  # Translation matrix
    p = R @ p + T
    
    # Plot
    h, = plt.plot(p[0, :], p[1, :], color=color, linewidth=2)
    
    return h

def drawprobellipse(x, C, alpha, color):
    """
    Draw elliptic probability region of a Gaussian in 2D.
    
    Parameters:
    x : array_like
        The [x, y, theta] ellipse center coordinates and rotation angle in radians.
        If theta is not provided, it will be calculated from the covariance matrix.
    C : array_like
        The 2x2 covariance matrix associated with the Gaussian distribution.
    alpha : float
        The significance level for the probability region.
    color : str or tuple
        The color specifier, either as a string or as an RGB tuple.
        
    Returns:
    h : matplotlib.lines.Line2D
        The handle to the line object representing the ellipse on the plot.
    """

    # Calculate unscaled half axes
    sxx, syy, sxy = C[0, 0], C[1, 1], C[0, 1]
    a = np.sqrt(0.5 * (sxx + syy + np.sqrt((sxx - syy)**2 + 4 * sxy**2)))  # always greater
    b = np.sqrt(0.5 * (sxx + syy - np.sqrt((sxx - syy)**2 + 4 * sxy**2)))  # always smaller

    # Remove imaginary parts in case of negative definite C
    a, b = np.real(a), np.real(b)

    # Scaling in order to reflect specified probability
    chi2_val = chi2.ppf(alpha, 2)  # Chi-squared inverse cumulative distribution function
    a *= np.sqrt(chi2_val)
    b *= np.sqrt(chi2_val)

    # Look where the greater half axis belongs to
    if sxx < syy:
        a, b = b, a

    # Calculate inclination (numerically stable)
    if sxx != syy:
        angle = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    elif sxy == 0:
        angle = 0  # angle doesn't matter
    elif sxy > 0:
        angle = np.pi / 4
    elif sxy < 0:
        angle = -np.pi / 4
    
    # If angle is not provided in x, set calculated angle
    if len(x) < 3:
        x = np.append(x, angle)

    # Draw ellipse
    h = drawellipse(x, a, b, color)
    
    return h

def compute_sigma_points(mu, sigma, lambda_, alpha, beta):
    """
    Sample sigma points from the distribution defined by mu and sigma according to the unscented transform.
    
    Parameters:
    mu : array_like
        Mean vector of the distribution.
    sigma : array_like
        Covariance matrix of the distribution.
    lambda_ : float
        Scaling parameter lambda.
    alpha : float
        Parameter to control the spread of the sigma points around mu.
    beta : float
        Parameter to incorporate prior knowledge of the distribution; for Gaussian x, beta=2 is optimal.
        
    Returns:
    sigma_points : ndarray
        Matrix of sigma points, each column represents one sigma point.
    w_m : ndarray
        Weight vector for the mean.
    w_c : ndarray
        Weight vector for the covariance.
    """
    n = len(mu)
    
    # Compute all sigma points
    sigma_points = np.zeros((n, 2*n + 1))
    # First sigma point is the mean
    sigma_points[:, 0] = mu.flatten()
    # Compute the square root matrix
    sqrt_matrix = sqrtm((n + lambda_) * sigma)
    for i in range(n):
        sigma_points[:, i + 1] = mu.flatten() + sqrt_matrix[:, i]
        sigma_points[:, n + i + 1] = mu.flatten() - sqrt_matrix[:, i]
    
    # Compute weight vectors for mean and covariance
    w_m = np.zeros(2*n + 1)
    w_c = np.zeros(2*n + 1)
    # Weights for the first sigma point
    w_m[0] = lambda_ / (n + lambda_)
    w_c[0] = w_m[0] + (1 - alpha**2 + beta)
    # Weights for the remaining sigma points
    w_m[1:] = 1 / (2 * (n + lambda_))
    w_c[1:] = w_m[1:]
    
    return sigma_points, w_m, w_c

def recover_gaussian(sigma_points, w_m, w_c):
    """
    Recover the Gaussian distribution (mean and covariance) given the sigma points and their weights.
    
    Parameters:
    sigma_points : ndarray
        Matrix of sigma points, where each column is one sigma point.
    w_m : array_like
        Weight vector for the mean.
    w_c : array_like
        Weight vector for the covariance.
        
    Returns:
    mu : ndarray
        The recovered mean vector of the distribution.
    sigma : ndarray
        The recovered covariance matrix of the distribution.
    """
    n = sigma_points.shape[0]

    # Compute the mean (mu)
    mu = np.sum(w_m * sigma_points, axis=1)

    # Compute the covariance (sigma)
    sigma = np.zeros((n, n))
    for i in range(2 * n + 1):
        diff = (sigma_points[:, i] - mu).reshape(n, 1)
        sigma += w_c[i] * diff @ diff.T

    return mu, sigma

def transform(points, function_number=1):
    """
    Apply a transformation to a set of points. Each column in 'points' is one point.

    Parameters:
    points : ndarray
        Matrix where each column is a point in the form [x; y].
    function_number : int
        The number of the function to apply.
        
    Returns:
    points : ndarray
        The transformed points.
    """

    if function_number == 1:
        # Function 1 (linear)
        # Applies a translation to [x; y]
        points[0, :] = points[0, :] + 1
        points[1, :] = points[1, :] + 2

    elif function_number == 2:
        # Function 2 (nonlinear)
        # Computes the polar coordinates corresponding to [x; y]
        x = points[0, :]
        y = points[1, :]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        points = np.vstack((r, theta))

    elif function_number == 3:
        # Function 3 (nonlinear)
        points[0, :] = points[0, :] * np.cos(points[0, :]) * np.sin(points[0, :])
        points[1, :] = points[1, :] * np.cos(points[1, :]) * np.sin(points[1, :])

    return points
