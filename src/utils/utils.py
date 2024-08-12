# Standard
import os
# External
import numpy as np
from PIL import Image
# Local


def make_gif(directory, output_filename='trajectory.gif', duration=2000):
    """
    Creates a GIF from all step_*.png files in the specified directory.

    Parameters
    ----------
    directory : str
        The directory containing the step_*.png files.
    output_filename : str, optional
        The name of the output GIF file, by default 'trajectory.gif'.
    duration : int, optional
        The duration of each frame in the GIF in milliseconds, by default 500ms.
    """
    # Get list of all step_*.png files in the directory and sort them
    images = sorted([img for img in os.listdir(directory) if img.startswith("step_") and img.endswith(".png")])

    # Ensure there are images to process
    if not images:
        raise ValueError("No step_*.png files found in the specified directory.")

    # Load images into a list
    frames = [Image.open(os.path.join(directory, img)) for img in images]

    # Save the frames as a GIF
    gif_path = os.path.join(directory, output_filename)
    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)

    print(f"GIF created at {gif_path}")

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

    # Handle the case when omega is zero (straight line motion)
    if omega == 0:
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = 0
    else:
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

def normalize_angle(phi):
    """
    Normalize an angle or an array of angles to the range [-pi, pi].
    
    This function takes an angle (in radians) or an array of angles and normalizes it to ensure it lies
    between -pi and pi. This is useful when working with angular measurements and 
    differences to avoid discontinuities due to angle wrapping.

    Parameters
    ----------
    phi : float or numpy.ndarray
        Angle (in radians) or an array of angles to be normalized.
    
    Returns
    -------
    float or numpy.ndarray
        Normalized angle or array of angles in the range [-pi, pi].
    """
    # If phi is a scalar, we normalize it directly
    if np.isscalar(phi) or len(phi)==1:
        while phi > np.pi:
            phi -= 2 * np.pi
        while phi < -np.pi:
            phi += 2 * np.pi
        return phi
    else:
        # If phi is an array, we apply the normalization element-wise
        normalized_phi = np.vectorize(normalize_angle)(phi)
        return normalized_phi

def normalize_all_bearings(z):
    """Go over the observations vector and normalize the bearings.
    The expected format of z is [range, bearing, range, bearing, ...]"""
    
    for i in range(1, len(z), 2):
        z[i] = normalize_angle(z[i])

    return z

def read_world(fpath):
    """Read world.dat file."""    
    data = {'ids': [], 'xy': []}
    
    with open(fpath, 'r') as file:
        for line in file.readlines():
            landmark_id, x, y = map(float, line.strip().split())
            data['ids'].append(int(landmark_id-1))
            data['xy'].append([x, y])
    
    return data

def read_data(fpath):
    """Read sensor_data.dat file."""    
    # Initialize the dictionary with empty lists
    data = {'timesteps': [], 'odometry': [], 'sensor': []}
    
    timestep = 0

    with open(fpath, 'r') as file:
        lines = file.readlines()
        i = 0
        
        while i < len(lines):
            line = lines[i].strip().split()
            
            if line[0] == "ODOMETRY":
                data['timesteps'].append(timestep)
                data['odometry'].append(list(map(float, line[1:])))
                i += 1  # Move to the next line
                
                # Add sensor data until the next ODOMETRY or end of file is reached
                sensors = []
                while i < len(lines):
                    line_elements = lines[i].strip().split()
                    if line_elements[0] == "SENSOR" and line_elements[1].isdigit():
                        sensor_id = int(line_elements[1]) - 1
                        sensor_data = [sensor_id] + list(map(float, line_elements[2:]))
                        sensors.append(sensor_data)
                        i += 1
                    else:
                        break
                data['sensor'].append(sensors)
                
                timestep += 1

    return data

