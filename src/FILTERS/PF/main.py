# Standard
import os
# External
import numpy as np
# Local
from .utils import Particle, Particle2D, resample
from .data import read_odometry_data
from .visualization import plot_state, plot_resampling
from .prediction import predict


def motion():
    # Assuming the read_data function is defined elsewhere to read odometry data
    odometry_data_path = os.path.join('exercises', '12_pf_framework', 'data', "odometry.dat")
    data = read_odometry_data(odometry_data_path)

    # Sensor noise
    noise = np.vstack([0.005, 0.01, 0.005])

    # How many particles
    num_particles = 100

    # Initialize a list of particles with equal weight and zero pose
    particles = [Particle(num_particles) for _ in range(num_particles)]

    # Perform filter update for each odometry-observation read from the data file
    for t, odom in enumerate(data):
        if t%100 == 0:
            print(f"Time: {t}")

        # Perform the prediction step of the particle filter
        particles = predict(particles, odom, noise)

        # Generate visualization plots of the current state of the filter
        plot_state(particles, t+1)

def resampling():
    # Initialize the particles
    num_particles = 1000
    particles = [Particle2D(num_particles) for _ in range(num_particles)]

    # Re-weight the particles according to their distance to [0, 0]
    sigma = np.diag([0.2, 0.2])
    sigma_inv = np.linalg.inv(sigma)
    for particle in particles:
        diff = particle.pose
        particle.weight = np.exp(-0.5 * diff.T @ sigma_inv @ diff).item()

    # Assuming the resample function is properly defined and available
    resampled_particles = resample(particles)

    # Plot the particles before and after resampling
    plot_resampling(particles, resampled_particles)
    

def main():
    motion()
    resampling()


if __name__ == "__main__":
    main()