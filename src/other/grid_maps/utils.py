# Standard
# External
import numpy as np
# Local

def log_odds_to_prob(l):
    """
    Convert log odds 'l' to the corresponding probability values 'p'.
    The input 'l' can be a scalar or a NumPy array.

    Parameters
    ----------
    l : scalar or numpy.ndarray
        The log odds to convert.

    Returns
    -------
    p : numpy.ndarray
        An array of probabilities corresponding to the log odds.
    """
    # Compute probability 'p' from log odds 'l' using the logistic function.
    p = 1 - 1 / (1 + np.exp(l))

    return p

def prob_to_log_odds(p):
    """
    Convert probability values 'p' to the corresponding log odds 'l'.
    The input 'p' can be a scalar or a NumPy array.

    Parameters
    ----------
    p : scalar or numpy.ndarray
        The probabilities to convert.

    Returns
    -------
    l : numpy.ndarray
        An array of log odds corresponding to the probabilities.
    """
    # Compute log odds 'l' from probability 'p'.
    l = np.log(p / (1 - p))

    return l

def world_to_map_coordinates(pntsWorld, gridSize, offset):
    """
    Convert points from the world coordinates frame to the map frame.

    Parameters
    ----------
    pntsWorld : np.ndarray
        An array where each column represents a point in world coordinates (meters).
    gridSize : float
        The size of each grid in meters.
    offset : np.ndarray
        The offset vector that needs to be subtracted from each point when converting to map coordinates.
    
    Returns
    -------
    np.ndarray
        A 2xN array containing the corresponding points in map coordinates.
    """
    # Compute map coordinates from world coordinates.
    pntsMap = np.floor((pntsWorld - np.tile(offset, (1, pntsWorld.shape[1]))) / gridSize)

    return pntsMap

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
    c = np.cos(v[2])
    s = np.sin(v[2])
    A = np.array([[c, -s, v[0]],
                  [s,  c, v[1]],
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
    v[0:2, 0] = A[0:2, 3]
    v[2, 0] = np.arctan2(A[1, 0], A[0, 1])
    return v

def robotlaser_as_cartesian(rl, maxRange=15, subsample=False):
    """
    Convert robot laser range readings to Cartesian coordinates.

    Parameters
    ----------
    rl : object
        A robot laser data structure with attributes 'ranges', 'maximum_range',
        'start_angle', 'angular_resolution', and 'laser_offset'.
    maxRange : float, optional
        The maximum range of laser readings to consider.
    subsample : bool, optional
        If True, subsample the beams by taking every other beam.

    Returns
    -------
    np.ndarray
        A 3xN array of points in Cartesian coordinates.
    """
    numBeams = len(rl.ranges)
    maxRange = min(maxRange, rl.maximum_range)
    # Apply the max range
    idx = [i for i, r in enumerate(rl.ranges) if maxRange > r > 0]

    if subsample:
        # Subsample the beams by removing every other index
        idx = idx[::2]

    angles = np.linspace(rl.start_angle, rl.start_angle + numBeams * rl.angular_resolution, numBeams)[idx]
    points = np.array([rl.ranges[i] * np.cos(angles[i]) for i in idx] +
                      [rl.ranges[i] * np.sin(angles[i]) for i in idx] +
                      [1 for _ in idx])

    # Apply the laser offset
    transf = v2t(rl.laser_offset)
    points = np.dot(transf, points)

    return points

def inv_sensor_model(map, scan, robPose, gridSize, offset, probOcc, probFree):
    """
    Compute the log odds values that should be added to the map based on the inverse sensor model
    of a laser range finder.

    Parameters
    ----------
    map : np.ndarray
        The matrix containing the occupancy values (IN LOG ODDS) of each cell in the map.
    scan : LaserScan
        A laser scan made at this time step, containing the range readings of each laser beam.
    robPose : np.ndarray
        The robot pose in the world coordinates frame.
    gridSize : float
        The size of each grid in meters.
    offset : np.ndarray
        The offset that needs to be subtracted from a point when converting to map coordinates.
    probOcc : float
        The probability that a cell is occupied by an obstacle given that a laser beam endpoint hit that cell.
    probFree : float
        The probability that a cell is occupied given that a laser beam passed through it.

    Returns
    -------
    mapUpdate : np.ndarray
        A matrix of the same size as map, with the log odds values that need to be added for the cells
        affected by the current laser scan. All unaffected cells should be zeros.
    robPoseMapFrame : np.ndarray
        The pose of the robot in the map coordinates frame.
    laserEndPntsMapFrame : np.ndarray
        The map coordinates of the endpoints of each laser beam (also used for visualization purposes).
    """
    # Initialize mapUpdate.
    mapUpdate = np.zeros_like(map)

    # Robot pose as a homogeneous transformation matrix.
    robTrans = v2t(robPose)

    # TODO: compute robPoseMapFrame using your world_to_map_coordinates implementation.
    robPoseMapFrame = world_to_map_coordinates(robPose[0:2], gridSize, offset)
    robPoseMapFrame = np.append(robPoseMapFrame, robPose[2])

    # Compute the Cartesian coordinates of the laser beam endpoints.
    laserEndPnts = robotlaser_as_cartesian(scan, 30, use_half_beams=False)

    # Compute the endpoints of the laser beams in the world coordinates frame.
    laserEndPnts = robTrans @ laserEndPnts
    # TODO: compute laserEndPntsMapFrame from laserEndPnts using your world_to_map_coordinates implementation.
    laserEndPntsMapFrame = world_to_map_coordinates(laserEndPnts[0:2], gridSize, offset)

    # Iterate over each laser beam and compute freeCells.
    freeCells = np.array([]).reshape(0, 2)
    for sc in range(laserEndPntsMapFrame.shape[1]):
        # Compute the XY map coordinates of the free cells along the laser beam ending in laserEndPntsMapFrame[:,sc]
        laserBeam = np.array([[robPoseMapFrame[0], robPoseMapFrame[1]], 
                              [laserEndPntsMapFrame[0, sc], laserEndPntsMapFrame[1, sc]]])
        X, Y = bresenham(laserBeam)
        # Add them to freeCells
        freeCells = np.vstack((freeCells, np.column_stack((X, Y))))

    # Update the log odds values in mapUpdate for each free cell according to probFree.
    for i in range(freeCells.shape[0]):
        x, y = int(freeCells[i, 0]), int(freeCells[i, 1])
        mapUpdate[x, y] = prob_to_log_odds(probFree)

    # Update the log odds values in mapUpdate for each laser endpoint according to probOcc.
    for i in range(laserEndPntsMapFrame.shape[1]):
        x, y = int(laserEndPntsMapFrame[0, i]), int(laserEndPntsMapFrame[1, i])
        mapUpdate[x, y] = prob_to_log_odds(probOcc)

    return mapUpdate, robPoseMapFrame, laserEndPntsMapFrame

def read_robotlaser(filename):
    """
    Read a file containing "ROBOTLASER1" in CARMEN logfile format.

    Parameters
    ----------
    filename : str
        The path to the file containing the laser data.

    Returns
    -------
    list
        A list of dictionaries, each containing data for a single laser reading.
    """
    laser = []
    with open(filename, 'r') as fid:
        while True:
            line = fid.readline()
            if line == '':
                break  # End of file
            tokens = line.split()
            if tokens[0] != "ROBOTLASER1":
                continue

            num_tokens = list(map(float, tokens))

            currentReading = {
                "start_angle": 0,
                "angular_resolution": 0,
                "maximum_range": 0,
                "ranges": [],
                "pose": np.zeros(3),
                "laser_offset": np.zeros(3),
                "timestamp": 0
            }

            tk = 2
            currentReading['start_angle'] = num_tokens[tk]
            tk += 2  # skip FOV
            currentReading['angular_resolution'] = num_tokens[tk]
            tk += 1
            currentReading['maximum_range'] = num_tokens[tk]
            tk += 3  # skip accuracy, remission_mode

            num_readings = int(num_tokens[tk])
            tk += 1
            currentReading['ranges'] = num_tokens[tk:tk+num_readings]
            tk += num_readings + 1  # skip remission values

            laser_pose = num_tokens[tk:tk+3]
            tk += 3
            robot_pose = num_tokens[tk:tk+3]
            tk += 3

            currentReading['laser_offset'] = t2v(np.linalg.inv(v2t(robot_pose)) @ v2t(laser_pose))
            
            tk += 5  # skip tv, rv, forward, side, turn
            currentReading['timestamp'] = num_tokens[tk]

            laser.append(currentReading)

    return laser

def bresenham(mycoords):
    """
    Generate a line profile using Bresenham's algorithm.

    Parameters
    ----------
    mycoords : list or np.ndarray
        List or array of coordinate pairs of the form: [x1, y1; x2, y2].

    Returns
    -------
    X, Y : tuple of lists
        The x and y coordinates of the line.
    """
    def swap(s, t):
        return t, s

    x = [round(val) for val in mycoords[:, 0]]
    y = [round(val) for val in mycoords[:, 1]]
    steep = abs(y[1] - y[0]) > abs(x[1] - x[0])

    if steep:
        x, y = swap(x, y)

    if x[0] > x[1]:
        x[0], x[1] = swap(x[0], x[1])
        y[0], y[1] = swap(y[0], y[1])

    delx = x[1] - x[0]
    dely = abs(y[1] - y[0])
    error = 0
    y_n = y[0]
    ystep = 1 if y[0] < y[1] else -1
    X, Y = [], []

    for x_n in range(x[0], x[1] + 1):
        if steep:
            X.append(y_n)
            Y.append(x_n)
        else:
            X.append(x_n)
            Y.append(y_n)
        error += dely
        if error << 1 >= delx:
            y_n += ystep
            error -= delx

    # Swap X and Y if the line is steep to maintain the order of x and y in the output
    if steep:
        X, Y = Y, X

    return X, Y
