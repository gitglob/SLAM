# Standard
import os
# External
import numpy as np
import matplotlib.pyplot as plt
# Local

def plot_state(particles, timestep):
    """
    Visualize the state of the particles.

    Parameters
    ----------
    particles : list of dicts
        Each dict contains the particle's pose and possibly other information.
    timestep : int
        The current timestep for the plot title.
    show_window : bool, optional
        If True, display the plot window; otherwise, save the plot to a file.
    """
    plt.clf()
    plt.grid(True)

    # Extract particle positions
    ppos = np.array([particle.pose[0:2] for particle in particles])

    # Plot the particles
    plt.plot(ppos[:, 0], ppos[:, 1], 'g.', markersize=10, linewidth=3.5)

    # Set plot limits
    plt.xlim([-2, 12])
    plt.ylim([-2, 12])

    # Save the plot
    directory = f'results/filters/PF/motion'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'{directory}/step_{timestep:03}.png'
    plt.savefig(filename)

    plt.close()

def plot_resampling(particles, resampled_particles):
    """
    Visualize the particles before and after resampling.

    Parameters
    ----------
    particles : list of Particle objects
        Each original particle.
    resampled_particles : list of Particle objects
        Easch particle after low variance resampling.
    """
    plt.clf()
    plt.grid(True)

    bpos = np.array([p.pose for p in particles])
    apos = np.array([p.pose for p in resampled_particles])

    plt.figure()
    plt.scatter(bpos[:, 0], bpos[:, 1], c='r', marker='+', s=5, label='Before Resampling')
    plt.scatter(apos[:, 0], apos[:, 1], c='b', marker='*', s=5, label='After Resampling')
    plt.legend()
    plt.grid(True)

    # Save the plot
    directory = f'results/filters/PF'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'{directory}/resampling.png'
    plt.savefig(filename)

    plt.close()