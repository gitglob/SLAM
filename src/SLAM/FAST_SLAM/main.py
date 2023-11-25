# Standard
import os
import time
# External
import numpy as np
# Local
from src.utils import read_world, read_data
from .utils import Particle, resample
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
    predict_times = []
    correct_times = []
    for t in range(data['timestep'].shape[1]):
        if t%100 == 0:
            print(f"Time: {t}")

        # Prediction step
        start_time = time.time()
        particles = predict(particles, data['timestep'][t]['odometry'], noise)
        predict_times.append(time.time() - start_time)

        # Correction step
        start_time = time.time()
        particles = correct(particles, data['timestep'][t]['sensor'])
        correct_times.append(time.time() - start_time)

        # Visualization
        plot_state(particles, landmarks, t, data['timestep'][t]['sensor'], show_gui)

        # Resample
        particles = resample(particles)

    print("FAST-SLAM finished!")

    print(f"Avg. prediction time: {sum(predict_times)/len(predict_times):.4f} sec.")
    print(f"Avg. correction time: {sum(correct_times)/len(correct_times):.4f} sec.")
    print(f"Avg. step time: {(sum(predict_times) + sum(correct_times))/(len(predict_times) + len(correct_times)):.4f} sec.")

    # Determine the best particle based on the highest weight
    best_particle_idx = np.argmax([p.weight for p in particles])
    best_particle = particles[best_particle_idx]
    print(f"Current state vector: \n{best_particle.pose}")
    # print(f"Current state covariance: {state_cov}")

if __name__ == "__main__":
    main()