# Standard
# External
import numpy as np
# Local


def velocityModel(state, u, dt):
    """
    Calculates the movement (displacement) of a robot based on the circular arc velocity model.
    
    Parameters:
    - state (list or array): A list containing the current state of the robot. 
                            [x, y, theta], where x and y are the position coordinates, 
                            and theta is the heading/orientation.
    - u (tuple or list): A tuple containing the control inputs [v, omega], 
                         where v is the linear velocity and omega is the yaw rate.
    - dt (float): Time step interval.
    
    Returns:
    - displacement (list): A list containing the displacement [dx, dy, dtheta], 
                           where dx and dy are the displacements in x and y directions, 
                           and dtheta is the change in heading/orientation.
    
    Raises:
    - ValueError: If omega is zero, as this leads to division by zero in the circular arc model.
    """
    # Extract linear velocity and yaw rate
    v, omega = u

    # Extract heading
    theta = state[2]

    # Noise term to not constrain the final orientation
    gamma = 1e-6

    if omega == 0:
        raise ValueError("Division by zero in the circular arc velocity model. ω=0.")
    
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
    xy_coords : list
        Shape will be either (N, 2) or (1, 2) corresponding to the input shape. 
        Contains the converted X and Y coordinates.
    """

    # Reshape input to ensure it's (N, 2)
    if polar_coords.shape == (2, 1):
        polar_coords = polar_coords.T

    r = polar_coords[:, 0]
    theta = polar_coords[:, 1]

    x_coords = r * np.cos(theta)
    y_coords = r * np.sin(theta)

    xy_coords = [[x, y] for (x, y) in zip(x_coords, y_coords)]

    return xy_coords

def xy2polar(xy_coords):
    """
    Convert x, y coordinates to range and yaw.

    Parameters
    ----------
    xy_coords : np.ndarray
        Shape can be either (N, 3) or (3, 1), where N is the number of data points. 
        Contains the X, Y coordinates.

    Returns
    -------
    polar_coords : list
        Shape will be either (N, 2) or (1, 2) corresponding to the input shape. 
        (:, 2) shape array with the converted range and yaw coordinates.
    """

    # Reshape input to ensure it's (N, 2)
    if xy_coords.shape == (3, 1):
        xy_coords = xy_coords.T
    
    range = np.sqrt(xy_coords[:, 0]**2 + xy_coords[:, 1]**2)
    bearing = np.arctan2(xy_coords[:, 1], xy_coords[:, 0])

    polar_coords = [[r, theta] for (r, theta) in zip(range, bearing)]

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
    # return angle % (2 * np.pi)
