# Standard
import os
# External
import numpy as np
from matplotlib import pyplot as plt
# Local
from src.visualization.plot_slam_state import drawprobellipse, drawrobot


def plot_state(particles, landmarks, timestep, z):
    """
    Visualizes the state of the FastSLAM algorithm.
    
    The function creates a plot displaying the map, particles, landmark estimates,
    and observations made at the current timestep.
    
    Parameters
    ----------
    particles : list of Particle
        The list of particles after applying the motion model.
    landmarks : list of Landmark
        The list of true landmark positions.
    timestep : int
        The current timestep in the simulation.
    z : list of Observations
        The observations made at the current timestep.
    window : bool
        If true, show the plot window; otherwise, save the plot to a file.
    """
    plt.clf()  # Clear the current figure.
    plt.grid(True)  # Add a grid to the plot.
    
    # Plot ground truth landmarks as black '+'
    for lm in landmarks["xy"]:
        plt.plot(lm[0], lm[1], 'k+', markersize=10, linewidth=5)
    
    # Plot particles in green
    ppos = np.array([p.pose for p in particles])
    plt.plot(ppos[:, 0], ppos[:, 1], 'g.')
    
    # Determine the best particle based on the highest weight
    best_particle_idx = np.argmax([p.weight for p in particles])
    best_particle = particles[best_particle_idx]
    
    # Draw the landmark locations and ellipses for the best particle
    for lm in best_particle.landmarks:
        if lm.observed:
            plt.plot(lm.mu[0], lm.mu[1], 'bo', markersize=3)
            drawprobellipse(lm.mu, lm.sigma, 0.95, 'b')
    
    # Draw observations as lines between robot and landmarks
    for obs in z:
        lm = best_particle.landmarks[obs[0]].mu
        plt.plot([best_particle.pose[0], lm[0]], [best_particle.pose[1], lm[1]], 'k-', linewidth=1)
    
    # Draw the trajectory of the best particle in red
    trajectory = np.array(best_particle.history)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2)
    
    # Draw the robot at the pose of the best particle
    drawrobot(best_particle.pose, 'r', 3, 0.3, 0.3)

    # Save the plot
    directory = f'results/slam/FAST'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'{directory}/step_{timestep:03}.png'
    plt.savefig(filename)

    plt.close()
    