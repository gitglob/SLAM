# Standard
# External
import numpy as np
# Local

def load_octave_cell_array(filename):
    """
    Load the laser data from the Octave text file.
    
    Highly customized to make it easier (the "laser" file has been modified).

    Parameters
    ----------
    filename : str
        The path to the Octave text file.

    Returns
    -------
    dict
        A dict with the list of values from the laser data.
    """

    # Load data
    with open(filename, 'r') as file:
        laser_dict = {}
        name = None
        for line in file:
            if line.startswith('# name: '):
                name = line[len('# name: '):].strip()
                laser_dict[name] = []
            elif line.startswith('# type: scalar'):
                line = next(file)
                laser_dict[name].append(float(line.strip()))
            elif name == "ranges" and line.startswith('# type: matrix'):
                line = next(file)
                line = next(file)
                line = next(file)
                matrix_line = line.strip()
                matrix_line = [float(x) for x in matrix_line.split()]
                laser_dict[name].append(matrix_line)
            elif name == "pose" and line.startswith('# type: matrix'):
                line = next(file)
                line = next(file)
                line = next(file)
                matrix_line = line.strip()
                matrix_line = [float(x) for x in matrix_line.split()]
                laser_dict[name].append(matrix_line)
            elif name == "laser_offset" and line.startswith('# type: matrix'):
                laser_offset = []
                line = next(file)
                line = next(file)
                line = next(file)
                laser_offset.append(float(line.strip()))
                line = next(file)
                laser_offset.append(float(line.strip()))
                line = next(file)
                laser_offset.append(float(line.strip()))
                laser_dict[name].append(laser_offset)

        # Convert lists of lists to numpy arrays
        laser_dict["start_angle"] = np.array(laser_dict["start_angle"]).reshape((-1,1))
        laser_dict["angular_resolution"] = np.array(laser_dict["angular_resolution"]).reshape((-1,1))
        laser_dict["maximum_range"] = np.array(laser_dict["maximum_range"]).reshape((-1,1))
        laser_dict["ranges"] = np.array(laser_dict["ranges"])
        laser_dict["ranges"] = laser_dict["ranges"][:, :, np.newaxis]
        laser_dict["pose"] = np.array(laser_dict["pose"])
        laser_dict["pose"] = laser_dict["pose"][:, :, np.newaxis]
        laser_dict["laser_offset"] = np.array(laser_dict["laser_offset"])
        laser_dict["laser_offset"] = laser_dict["laser_offset"][:, :, np.newaxis]
        laser_dict["timestamp"] = np.array(laser_dict["timestamp"])

    return laser_dict