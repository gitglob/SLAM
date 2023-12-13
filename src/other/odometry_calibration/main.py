# Standard
import os
import time
# External
import numpy as np
import matplotlib.pyplot as plt
# Local
from .utils import ls_calibrate_odometry, compute_trajectory, apply_odometry_correction
from .data import load_odom


def main():
    # Load the odometry and scan-matched motions data
    # Assuming these are loaded from files or defined elsewhere
    odom_motions = load_odom("exercises/14_odom_calib_framework/data/odom_motions")
    scanmatched_motions = load_odom("exercises/14_odom_calib_framework/data/scanmatched_motions")

    # Create our measurements vector z
    z = np.hstack([scanmatched_motions, odom_motions])

    # Perform the calibration
    X = ls_calibrate_odometry(z)
    print('Calibration result:\n', X)

    # Apply the estimated calibration parameters
    calibrated_motions = apply_odometry_correction(X, odom_motions)

    # Compute the trajectories
    odom_trajectory = compute_trajectory(odom_motions)
    scanmatch_trajectory = compute_trajectory(scanmatched_motions)
    calibrated_trajectory = compute_trajectory(calibrated_motions)
   
    # Save the plot
    directory = f'results/other/odom_calib'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'{directory}/odometry-calibration.png'

    # Plot the trajectories
    plt.plot(odom_trajectory[:, 0], odom_trajectory[:, 1], label="Uncalibrated Odometry")
    plt.plot(scanmatch_trajectory[:, 0], scanmatch_trajectory[:, 1], label="Scan-Matching")
    plt.plot(calibrated_trajectory[:, 0], calibrated_trajectory[:, 1], label="Calibrated Odometry")
    plt.legend()
    plt.savefig(filename)


if __name__ == "__main__":
    main()
    