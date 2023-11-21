# Standard
import os
# External
import numpy as np
# Local

def load_octave_cell_array(filename):
    """
    Load a cell array from an Octave text file.

    Parameters
    ----------
    filename : str
        The path to the Octave text file.

    Returns
    -------
    dict
        A dict with the list of values from the laser data.
    """

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

    return laser_dict