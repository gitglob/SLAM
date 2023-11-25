# Standard
from copy import deepcopy
# External
import numpy as np
# Local
from src.utils import normalize_angle

class Particle:
    """2D Particle class."""
    def __init__(self, num_particles):
        self.weight = 1.0 / num_particles
        self.pose = np.zeros((3,1))
        self.history = []

class Particle2D:
    """3D Particle class."""
    def __init__(self, num_particles):
        self.weight = 1.0 / num_particles
        self.pose = np.empty((2,1))
        self.pose[0] = np.random.normal(0, 1)
        self.pose[1] = np.random.normal(0, 2)
        self.history = []

def low_variance_resampling(particles, weights, num_particles):
    new_particles = []
    c = weights[0]
    r = np.random.uniform(0, 1 / num_particles)

    # Walk along the wheel to select the particles
    i = 0
    for j in range(num_particles):
        U = r + (j-1) / num_particles
        while U > c:
            i += 1
            c += weights[i]
        
        # Add partricle i to the list of new particles
        new_particles.append(deepcopy(particles[i]))

    return new_particles

def resample(particles):
    """
    Resamples the set of particles using the low variance resampling method.

    Parameters
    ----------
    particles : list of Particle
        The list of particles representing the belief about the robot's position.

    Returns
    -------
    list of Particle
        The list of resampled particles.
    """
    num_particles = len(particles)
    weights = np.array([particle.weight for particle in particles])

    # Normalize the weights
    weights /= np.sum(weights)

    # Check number of effective particles, to decide whether to resample or not
    use_neff = False
    if use_neff:
        neff = 1. / np.sum(weights ** 2)
        if neff > 0.5 * num_particles:
            # If the effective number of particles is more than half of the total number
            # no resampling is done.
            newParticles = deepcopy(particles)
            for i in num_particles:
                newParticles[i].weight = weights[i]

            return particles

    # Low variance re-sampling
    new_particles = low_variance_resampling(particles, weights, num_particles)

    return new_particles

def measurement_model(particle, z):
    """
    Compute the expected measurement for a landmark and the Jacobian with respect to the landmark.

    Parameters
    ----------
    particle : Particle
        The particle representing a possible state of the robot.
    z : Observation
        The observation of a landmark.

    Returns
    -------
    h : numpy.ndarray
        The expected measurement.
    H : numpy.ndarray
        The Jacobian of the measurement model.
    """
    # Extract the landmark position from the particle's estimated map
    landmark_id = z.id
    landmark_pos = particle.landmarks[landmark_id].mu

    # Calculate the expected range and bearing
    dx = landmark_pos[0] - particle.pose[0]
    dy = landmark_pos[1] - particle.pose[1]
    expected_range = np.sqrt(dx**2 + dy**2)
    expected_bearing = normalize_angle(np.arctan2(dy, dx) - particle.pose[2])
    h = np.array([expected_range, expected_bearing])

    # Compute the Jacobian matrix H of the measurement function h with respect to the landmark position
    H = np.zeros((2, 2))
    H[0, 0] = dx / expected_range
    H[0, 1] = dy / expected_range
    H[1, 0] = -dy / (expected_range**2)
    H[1, 1] = dx / (expected_range**2)

    return h, H
