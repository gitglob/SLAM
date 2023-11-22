# Standard
# External
import numpy as np
# Local
from .utils import prob_to_log_odds, inv_sensor_model
from .visualize import plot_map
from .data import load_octave_cell_array


def main():
    # Load laser scans and robot poses.
    laser = load_octave_cell_array("exercises/12_gridmaps_framework/data/laser_copy")
    poses = laser["pose"]

    # Initial cell occupancy probability.
    prior = 0.50
    # Probabilities related to the laser range finder sensor model.
    probOcc = 0.9
    probFree = 0.35

    # Map grid size in meters. Decrease for better resolution.
    gridSize = 0.1

    # Set up map boundaries and initialize map.
    border = 30
    robXMin = np.min(poses[:, 0])
    robXMax = np.max(poses[:, 0])
    robYMin = np.min(poses[:, 1])
    robYMax = np.max(poses[:, 1])
    mapBox = [robXMin - border, robXMax + border, robYMin - border, robYMax + border]
    offsetX = mapBox[0]
    offsetY = mapBox[2]
    mapSizeMeters = [mapBox[1] - offsetX, mapBox[3] - offsetY]
    mapSize = np.ceil(np.array(mapSizeMeters) / gridSize).astype(int)

    # Convert probabilities to log odds
    logOddsPrior = prob_to_log_odds(prior)

    # The occupancy value of each cell in the map is initialized with the prior.
    map = logOddsPrior * np.ones(mapSize)
    print('Map initialized. Map size:'), print(map.shape)

    # Map offset used when converting from world to map coordinates.
    offset = np.vstack([offsetX, offsetY])

    # Main loop for updating map cells.
    for t in range(poses.shape[0]):
        if t%100 == 0:
            print(f"Time: {t}")

        # Robot pose at time t.
        robPose = poses[t]

        # Laser scan made at time t.
        scan = {}
        for key in laser.keys():
            scan[key] = laser[key][t]

        # Compute the mapUpdate, which contains the log odds values to add to the map.
        mapUpdate, robPoseMapFrame, laserEndPntsMapFrame = inv_sensor_model(map, scan, robPose, gridSize, offset, probOcc, probFree)

        mapUpdate -= logOddsPrior * np.ones_like(map)
        # Update the occupancy values of the affected cells.
        map += mapUpdate

        # Plot current map and robot trajectory so far.
        plot_map(map, robPoseMapFrame, poses, laserEndPntsMapFrame, gridSize, offset, t)

if __name__ == "__main__":
    main()