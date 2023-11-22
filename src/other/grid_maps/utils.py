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
    # Compute probability 'p' from log odds 'l' using the logistic function
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
    # Compute log odds 'l' from probability 'p'
    l = np.log(p / (1 - p))

    return l

def world_to_map_coordinates(pntsWorld, gridSize, offset):
    """
    Convert points from world coordinates to map coordinates.

    Parameters:
    ----------
    pntsWorld : np.ndarray
        a 2x1 or Nx2x1 array of points in world coordinates (meters)
    gridSize : np.ndarray 
        the size of each grid in meters
    offset : np.ndarray 
        a a 2x1 array [offsetX, offsetY] representing the offset to be subtracted

    Returns:
    --------
    pntsMap : np.ndarray 
        an array of the same shape as pntsWorld containing the corresponding points in map coordinates
    """
    # Ensure the offset is an array for broadcasting
    offset = np.array(offset)

    # Align points to the map origin, scale points to the new grid size and round every number down to the closes integer
    pntsMap = np.floor((pntsWorld - offset) / gridSize)
    pntsMap = np.squeeze(pntsMap, axis=-1)

    return pntsMap.astype(int)

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
    tx = v[0].item()
    ty = v[1].item()
    c = np.cos(v[2]).item()
    s = np.sin(v[2]).item()
    A = np.array([[c, -s, tx],
                  [s,  c, ty],
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
    v[0:2, 0] = A[0:2, 2]
    v[2, 0] = np.arctan2(A[1, 0], A[0, 0])
    return v

def robotlaser_as_cartesian(scan, maxRange=15):
    """
    Convert robot laser range readings to Cartesian coordinates.

    Parameters
    ----------
    scan : object
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
    numBeams = len(scan["ranges"])
    maxRange = min(maxRange, scan["maximum_range"])
    # Apply the max range
    idx = [i for i, r in enumerate(scan["ranges"]) if maxRange > r > 0]

    angles = np.linspace(scan["start_angle"], scan["start_angle"] + numBeams * scan["angular_resolution"], numBeams)[idx]
    ranges = scan["ranges"][idx]
    points = np.hstack((ranges * np.cos(angles),
                        ranges * np.sin(angles),
                        np.ones((len(angles), 1))))

    # Apply the laser offset
    transf = v2t(scan["laser_offset"])
    points = (transf @ points.T).T

    return points

def swap(s, t):
    # Swap variable values
    return t, s

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

    x = [round(p) for p in (mycoords[0][0], mycoords[1][0])]
    y = [round(p) for p in (mycoords[0][1], mycoords[1][1])]
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
        if 2 * error >= delx:
            y_n += ystep
            error -= delx

    return X, Y

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
    # Initialize mapUpdate
    mapUpdate = np.zeros_like(map)

    # Robot pose as a homogeneous transformation matrix
    robTrans = v2t(robPose)

    # Compute the robot pose in the map
    robPoseMapFrame = world_to_map_coordinates(robPose[0:2], gridSize, offset)
    robPoseMapFrame = np.append(robPoseMapFrame, robPose[2])

    # Compute the Cartesian coordinates of the laser beam endpoints
    laserEndPnts = robotlaser_as_cartesian(scan, 30)

    # Compute the endpoints of the laser beams in the world coordinates frame
    laserEndPnts = (robTrans @ laserEndPnts.T).T
    laserEndPnts = laserEndPnts[:, :, np.newaxis]
    
    # Compute the laser end points in the map frame
    laserEndPntsMapFrame = world_to_map_coordinates(laserEndPnts[:, 0:2], gridSize, offset)

    # Iterate over each laser beam and compute freeCells
    freeCells = np.array([]).reshape(0, 2)
    for sc in range(laserEndPntsMapFrame.shape[0]):
        # Compute the XY map coordinates of the free cells along the laser beam ending
        laserBeam = np.array([[robPoseMapFrame[0].item(), robPoseMapFrame[1].item()],
                              [laserEndPntsMapFrame[sc, 0].item(), laserEndPntsMapFrame[sc, 1].item()]])
        X, Y = bresenham(laserBeam)

        # Add them to freeCells
        freeCells = np.vstack((freeCells, np.column_stack((X, Y))))

    # Update the log odds values in mapUpdate for each free cell
    for i in range(freeCells.shape[0]):
        x, y = int(freeCells[i, 0]), int(freeCells[i, 1])
        mapUpdate[x, y] = prob_to_log_odds(probFree)

    # Update the log odds values in mapUpdate for each laser endpoint
    for i in range(laserEndPntsMapFrame.shape[0]):
        x, y = int(laserEndPntsMapFrame[i, 0]), int(laserEndPntsMapFrame[i, 1])
        mapUpdate[x, y] = prob_to_log_odds(probOcc)

    return mapUpdate, robPoseMapFrame, laserEndPntsMapFrame
