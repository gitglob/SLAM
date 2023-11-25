# Standard
# External
import numpy as np
# Local


def read_odometry_data(filename):
    """
    Read odometry data from a file and return it as an Nx3 NumPy array.
    
    Each line in the file should start with 'ODOMETRY' followed by three
    floating-point numbers representing odometry measurements.
    
    Parameters:
    filename : str
        The path to the file containing odometry data.
        
    Returns:
    data : ndarray
        An Nx3 array where N is the number of odometry entries in the file,
        and each row is of the form [x, y, theta].
        
    Raises:
    IOError: If the file cannot be read.
    ValueError: If the data format in the file is incorrect.
    """
    
    # Initialize an empty list to store the data
    data_list = []
    
    # Open the file for reading
    with open(filename, 'r') as file:
        # Read each line in the file
        for line in file:
            # Split the line into words
            words = line.split()
            # Check if the line starts with 'ODOMETRY'
            if words[0] == 'ODOMETRY':
                # Extract the three odometry measurements and convert to float
                measurements = [float(value) for value in words[1:4]]
                # Append to the data list
                data_list.append(measurements)
            else:
                raise ValueError(f"Unexpected format in line: {line}")
    
    # Convert the list of data to a NumPy array
    data = np.array(data_list)
    
    return data