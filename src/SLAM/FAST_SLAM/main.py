# Standard
import os
import time
# External
import numpy as np
import matplotlib.pyplot as plt
# Local
from src.utils import read_world, read_data
from .utils import Landmark, Particle, resample
from .visualization import plot_state
from .prediction import predict
from .correction import correct


def main():
    # Read world and sensor data
    world_path = os.path.join('exercises', '13_fastslam_framework', 'data', "world.dat")
    landmarks = read_world(world_path)
    sensor_path = os.path.join('exercises', '13_fastslam_framework', 'data', "sensor_data.dat")
    data = read_data(sensor_path)

    # Get the number of landmarks in the map
    num_landmarks = landmarks.shape[1]

    noise = np.vstack([0.005, 0.01, 0.005])

    # Number of particles
    num_particles = 100

    # Initialize the particles array
    particles = [Particle(num_landmarks, num_particles) for _ in range(num_particles)]

    # Toggle the visualization type
    show_gui = False

    # Main loop
    for t in range(data['timestep'].shape[1]):
        print(f'timestep = {t}')

        # Prediction step
        particles = predict(particles, data['timestep'][t]['odometry'], noise)

        # Correction step
        particles = correct(particles, data['timestep'][t]['sensor'])

        # Visualization
        plot_state(particles, landmarks, t, data['timestep'][t]['sensor'], show_gui)

        # Resample
        particles = resample(particles)


if __name__ == "__main__":
    main()