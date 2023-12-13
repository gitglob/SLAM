# Standard
# External
import numpy as np
# Local

def load_odom(filename):
    """
    Reads odometry data from a file and returns it as a NumPy array.

    Parameters
    ----------
    filename : str
        Path to the file containing the odometry data.

    Returns
    -------
    numpy.ndarray
        A Numpy array of shape (N, 3), where each row represents [x, y, theta].
    """
    data = []
    with open(filename, 'r') as file:
        i = 0
        for line in file:
            i+=1
            # Skip lines that do not contain numerical data
            if line.startswith('#') or line.isspace():
                continue
            # Split the line into values and convert to float
            values = [float(val) for val in line.split()]
            data.append(values)
    
    return np.array(data)
