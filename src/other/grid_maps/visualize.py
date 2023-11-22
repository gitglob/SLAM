# Standard
import os
# External
import numpy as np
import matplotlib.pyplot as plt
# Local
from .utils import log_odds_to_prob, world_to_map_coordinates

def plot_map(map, robPoseMapFrame, poses, laserEndPntsMapFrame, gridSize, offset, t):
    """
    Visualize the robot's occupancy grid map, trajectory, pose, and laser scan endpoints.

    Parameters
    ----------
    map : np.ndarray
        The occupancy grid map in log-odds format.
    mapBox : list
        The map boundaries.
    robPoseMapFrame : list
        The robot's pose in the map frame.
    poses : np.ndarray
        The robot's trajectory poses.
    laserEndPntsMapFrame : np.ndarray
        The endpoints of the robot's laser scans in map frame.
    gridSize : float
        The size of each grid cell in the map.
    offset : np.ndarray
        The offset applied to the map coordinates.
    t : int
        The current time step or iteration number.
    """
    poses = np.squeeze(poses, -1)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.imshow(1 - log_odds_to_prob(np.transpose(map)), cmap='gray')

    traj_poses = np.column_stack((poses[:t+1, 0], poses[:t+1, 1]))
    traj_poses = np.expand_dims(traj_poses, axis=-1)
    traj = world_to_map_coordinates(traj_poses, gridSize, offset)
    
    ax.plot(traj[:, 0], traj[:, 1], 'g')
    ax.plot(robPoseMapFrame[0], robPoseMapFrame[1], 'bo', markersize=5, linewidth=4)
    ax.plot(laserEndPntsMapFrame[:, 0], laserEndPntsMapFrame[:, 1], 'ro', markersize=2)
    
    directory = f"results/other/gridmaps"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{directory}/gridmap_{t:03}.png"

    plt.savefig(filename)
    plt.close("all")
