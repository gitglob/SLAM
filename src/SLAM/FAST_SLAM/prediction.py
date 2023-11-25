# Standard
# External
import numpy as np
# Local
from src.utils import normalize_angle

def predict(particles, u, noise):
    """
    Performs the prediction step of the FastSLAM algorithm on all particles.
    
    The function updates the particles by sampling from the motion model,
    which is perturbed by Gaussian noise.
    
    Parameters
    ----------
    particles : list of Particle
        The list of particles representing the belief about the robot's position.
    u : dict
        The control input containing rotation and translation values (`r1`, `t`, `r2`).
    noise : numpy.ndarray
        The standard deviations of the Gaussian noise for the motion model.
        
    Returns
    -------
    list of Particle
        The list of particles after applying the motion model.
    """
    # Extract noise parameters for r1, translation, and r2 from the noise array
    r1_noise, trans_noise, r2_noise = noise
    
    # Update each particle
    for particle in particles:
        # Append the old position to the history of the particle
        particle.history.append(particle.pose.copy())
        
        # Sample a new pose for the particle using Gaussian noise
        r1 = np.random.normal(u['r1'], r1_noise)
        r2 = np.random.normal(u['r2'], r2_noise)
        trans = np.random.normal(u['t'], trans_noise)
        
        # Update the x and y position based on the sampled translation and rotation
        particle.pose[0] += trans * np.cos(particle.pose[2] + r1)
        particle.pose[1] += trans * np.sin(particle.pose[2] + r1)

        # Normalize the angle to ensure it stays within the range [-pi, pi]
        particle.pose[2] = normalize_angle(particle.pose[2] + r1 + r2)
        
    return particles
