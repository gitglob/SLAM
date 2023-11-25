# Standard
# External
import numpy as np
# Local
from src.utils import normalize_angle


def predict(particles, u, noise):
    """
    Update the particles by sampling from the motion model with added Gaussian noise.

    Parameters
    ----------
    particles : list of dicts
        Each dict contains the particle's pose and history.
    u : dict
        A dict containing the control actions (r1, t, r2).
    noise : numpy.ndarray
        The standard deviations of the Gaussian noise for the motion model.

    Returns
    -------
    list of dicts
        The updated list of particles with new poses.
    """
    # Unpack noise parameters
    r1_noise, trans_noise, r2_noise = noise

    # Update each particle
    for particle in particles:
        # Append the old position to the history of the particle
        particle.history.append(particle.pose.copy())

        # Sample a new pose for the particle
        r1 = np.random.normal(u[0], r1_noise)
        trans = np.random.normal(u[1], trans_noise)
        r2 = np.random.normal(u[2], r2_noise)

        # Update particle pose
        new_x = particle.pose[0] + trans * np.cos(particle.pose[2] + r1)
        new_y = particle.pose[1] + trans * np.sin(particle.pose[2] + r1)
        new_theta = particle.pose[2] + r1 + r2

        # Normalize the angle to ensure it stays within the range [-pi, pi]
        new_theta = normalize_angle(new_theta)

        # Assign the new pose to the particle
        particle.pose = np.array([new_x, new_y, new_theta])

    return particles