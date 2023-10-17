# Standard
import os
# External
import numpy as np
import matplotlib.pyplot as plt
# Local


def plot_filter_trajectories(robot_states, robot_times, ground_truth_states, ground_truth_times, filter_name):
    """
    Plots and saves the trajectories of the robot and its ground truth.

    Parameters
    ----------
    robot_states : np.ndarray
        States of the robot in the format (x, y, theta).
    robot_times : np.ndarray
        Timestamps corresponding to the robot_states.
    ground_truth_states : np.ndarray
        Ground truth states in the format (x, y, theta).
    ground_truth_times : np.ndarray
        Timestamps corresponding to the ground_truth_states.
    destination : str
        The sub-directory within 'results' where the plot will be saved.

    Returns
    -------
    None
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot robot trajectory
    ax.plot(robot_states[:, 0], robot_states[:, 1], 'b-', label='Robot Trajectory', alpha=0.7)
    
    # Plot ground truth trajectory
    ax.plot(ground_truth_states[:, 0], ground_truth_states[:, 1], 'r--', label='Ground Truth', alpha=0.7)
    
    ax.set_title('Robot and Ground Truth Trajectories')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)
    
    # Check if the 'results' folder exists
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Save the plot to the specified destination within 'results'
    save_path = os.path.join("results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"{filter_name}-trajectory_comparison.png"))
    plt.close()
