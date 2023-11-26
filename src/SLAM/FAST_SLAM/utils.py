# Standard
from copy import deepcopy
# External
import numpy as np
# Local

class Landmark:
    """Landmark class."""
    def __init__(self):
        self.observed = False
        self.mu = np.zeros((2,1))  # 2D position
        self.sigma = np.zeros((2, 2))  # covariance

class Particle:
    """Particle class."""
    def __init__(self, num_landmarks, num_particles):
        self.weight = 1.0 / num_particles
        self.pose = np.zeros((3,1))
        self.history = []
        self.landmarks = [Landmark() for _ in range(num_landmarks)]

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
            new_particles = deepcopy(particles)
            for i in num_particles:
                new_particles[i].weight = weights[i]

            return particles

    # Implement low variance re-sampling
    new_particles = []
    cs = np.cumsum(weights)
    weight_sum = cs[len(cs)-1]

    # initialize the step and the current position on the roulette wheel
    step = weight_sum / num_particles
    position = np.random.uniform(0, weight_sum)

    # Walk along the wheel to select the particles
    idx = 0
    for particle in particles:
        position += step

        if position > weight_sum:
            position -= weight_sum
            idx = 0

        while position > cs[idx]:
            idx += 1

        # Create a new particle and add it to the list of new particles
        new_particle = deepcopy(particle)
        new_particle.weight = 1.0 / num_particles
        new_particles.append(new_particle)

    return new_particles
