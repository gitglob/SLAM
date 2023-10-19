# Standard
# External
import numpy as np
# Local

def velocityModel(state, u, dt):
    """Calculates the movement of the robot (displacement) based on the circular arc velocity model."""
    # Extract linear velocity and yaw rate
    v, omega = u

    # Extract heading
    theta = state[2]

    # Noise term to not constrain the final orientation
    gamma = 1e-6

    # Displacement from the velocity model - circular arc model
    dx = (-v/omega) * np.sin(theta) + (v/omega) * np.sin(theta + omega*dt)
    dy = (v/omega) * np.cos(theta) - (v/omega) * np.cos(theta + omega*dt)
    dtheta = omega*dt #+ gamma*dt
    displacement = [dx, dy, dtheta]

    return displacement

def polar2xy(polar_coords):
    """
    Convert range and yaw data to x, y coordinates.

    Parameters
    ----------
    polar_coords : np.ndarray
        Shape can be either (N, 2) or (2, 1), where N is the number of data points. 
        Contains range and yaw coordinates.

    Returns
    -------
    xy_coords : np.ndarray
        Shape will be either (N, 2) or (2, 1) corresponding to the input shape. 
        Contains the converted X and Y coordinates.
    """

    # Reshape input to ensure it's (N, 2)
    if polar_coords.shape == (2, 1):
        polar_coords = polar_coords.T

    r = polar_coords[:, 0]
    theta = polar_coords[:, 1]

    x_coords = r * np.cos(theta)
    y_coords = r * np.sin(theta)

    xy_coords = [x_coords, y_coords]

    return xy_coords

def xy2polar(xy_coords):
    """
    Convert x, y coordinates to range and yaw.

    Parameters
    ----------
    xy_coords : np.ndarray
        (:, 2) array with X, Y coordinates.

    Returns
    -------
    polar_coords : np.ndarray()
        (:, 2) shape array with the converted range and yaw coordinates.
    """
    
    r = np.sqrt(xy_coords[:, 0]**2 + xy_coords[:, 1]**2)
    theta = np.arctan2(xy_coords[:, 1], xy_coords[:, 0])

    polar_coords = np.column_stack((r, theta))

    return polar_coords

def normalize_angle(angle):
    """
    Normalize an angle to the range [-pi, pi].
    
    This function takes an angle (in radians) and normalizes it to ensure it lies
    between -pi and pi. This is useful when working with angular measurements and 
    differences to avoid discontinuities due to angle wrapping.

    Parameters
    ----------
    angle : float
        Angle (in radians) to be normalized.
    
    Returns
    -------
    float
        Normalized angle in the range [-pi, pi].
    """
    return np.arctan2(np.sin(angle), np.cos(angle))
