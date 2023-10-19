# Standard
import os
# External
import numpy as np
import matplotlib.pyplot as plt
# Local
from utils import polar2xy, velocityModel
from . import range_noise_std, yaw_noise_std, sensor_frequency
from . import timesteps, timestep_duration, initial_radius, growth_factor


def simulate_spiral_movement(timesteps=timesteps, timestep_duration=timestep_duration, 
                             initial_radius=initial_radius, growth_factor=growth_factor):
    """
    Simulates a spiral movement for a vehicle based on the circular arc velocity model.
    
    The idea behind the simulation is to use the vehicle's velocity and yaw rate
    to generate a spiral motion. At each timestep, the desired radius of the spiral
    is calculated. The vehicle's velocity and yaw rate are then adjusted to achieve
    this radius, and the vehicle's position is updated based on the velocity model.

    Returns
    -------
    x, y, theta, t, v, omega : np.ndarray, ...
        - X and Y coordinates of the vehicle over time (m).
        - Time corresponding to the coordinates (sec).
        - Heading (radians) over time.
        - Velocity (m/s) over time.
        - Yaw rate (rad/s) over time.
    """

    # Initialize lists to store trajectory data
    x = [0]
    y = [0]
    theta = [np.pi/2]  # Start pointing upwards, which is pi/2 radians from the x-axis
    
    # Generate a linear time array based on the specified timesteps and duration
    t = np.linspace(0, timesteps*timestep_duration, timesteps)

    # Define a constant linear velocity for the vehicle
    v = 10.0

    # Keep vehicle rotational velocity
    omegas = [0]
    
    for i in range(1, timesteps):
        # Calculate the desired spiral radius based on the elapsed time and growth factor.
        # This is a simple linear model where the radius grows over time.
        current_radius = initial_radius + growth_factor * t[i]
        
        # Calculate the vehicle's velocity and yaw rate to achieve the desired spiral.
        # The idea is to adjust the speed and turning rate to maintain the growth 
        # of the spiral as determined by the current_radius.
        omega = v / current_radius
        omegas.append(omega)

        # Fetch the previous state of the robot (position and heading).
        state = [x[-1], y[-1], theta[-1]]
        
        # Use the velocity model to calculate how much the robot moves during the current timestep.
        velocity_profile = ([v, omega])
        displacement = velocityModel(state, velocity_profile, timestep_duration)

        # Update the robot's position and heading based on the calculated displacement.
        x.append(x[-1] + displacement[0])
        y.append(y[-1] + displacement[1])
        theta.append(theta[-1] + displacement[2])

    # Convert lists to numpy arrays for consistency and easier processing
    return np.array(x), np.array(y), np.array(theta), t, np.full_like(t, v), np.array(omegas)

def simulate_sensors(x, y, t, add_noise=False,
                     range_noise_std=range_noise_std, yaw_noise_std=yaw_noise_std, 
                     sensor_frequency=sensor_frequency):
    """
    Simulates two sensors: one measuring range and another measuring yaw.
    
    Parameters
    ----------
    x, y : np.ndarray, np.ndarray
        X and Y coordinates of the vehicle.
    t : np.ndarray
        Time vector corresponding to the X and Y coordinates.
    range_noise_std : float
        Standard deviation of noise for the range sensor.
    yaw_noise_std : float
        Standard deviation of noise for the yaw sensor.
    range_frequency : float
        Frequency of the range measurements. A value greater than 1 means multiple readings per second,
        a value less than 1 means readings spread over multiple seconds.
    yaw_frequency : float
        Frequency of the yaw measurements. A value greater than 1 means multiple readings per second,
        a value less than 1 means readings spread over multiple seconds.

    Returns
    -------
    range_data, yaw_data : (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)
        Tuples of range and yaw measurements and their associated timestamps.
    """
    
    # Determine timesteps for measurements based on frequency
    sensor_t = np.arange(t[0], t[-1], 1/sensor_frequency)
    sensor_indices = [np.argmin(np.abs(t - time)) for time in sensor_t]
    
    # Extract range and yaw values at the specified timesteps and add measurement noise
    range_measurements = np.sqrt(x[sensor_indices]**2 + y[sensor_indices]**2) 
    yaw_measurements = np.arctan2(y[sensor_indices], x[sensor_indices])

    if add_noise:
        range_measurements += np.random.normal(0, range_noise_std, len(sensor_indices))
        yaw_measurements += np.random.normal(0, yaw_noise_std, len(sensor_indices))

    sensor_measurements = np.vstack((range_measurements, yaw_measurements)).T
    
    return (sensor_measurements, sensor_t)

def plot_results(x, y, sensor_measurements, sensor_ts):
    """
    Plots the spiral movement of the robot and its sensor readings.

    Parameters
    ----------
    x, y : np.ndarray, np.ndarray
        X and Y coordinates of the vehicle.
    sensor_measurements, sensor_ts : np.ndarray, np.ndarray
        Range and yaw measurements with their respective timestamp.

    Returns
    -------
    None
    """
    # Convert range and yaw data back to x and y
    sensor_xy = polar2xy(sensor_measurements)

    fig, ax = plt.subplots(3, 1, figsize=(8, 15))
    
    # Spiral movement plot
    ax[0].plot(x, y,linewidth=.5, label='Ground Truth Path', color='blue')
    ax[0].scatter(sensor_xy[:,0], sensor_xy[:,1], c='red', s=5, label='Sensor Readings', marker='x')
    ax[0].set_title('Spiral Movement of the Robot')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].grid(True)
    ax[0].axis('equal')
    ax[0].legend()

    # Range sensor readings plot
    ax[1].plot(sensor_ts, sensor_measurements[:,0], label='Range Readings', color='green')
    ax[1].set_title('Range Readings over Time')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Range (m)')
    ax[1].grid(True)
    ax[1].legend()

    # Yaw sensor readings plot
    ax[2].plot(sensor_ts, sensor_measurements[:,1], label='Yaw Readings', color='orange')
    ax[2].set_title('Yaw Readings over Time')
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Yaw (rad)')
    ax[2].grid(True)
    ax[2].legend()

    plt.tight_layout()
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig('results/simulated_trajectory.png', dpi=300)


def main():
    # Simulate spiral movement
    x, y, theta, t, v, omega = simulate_spiral_movement()

    # Simulate sensor readings
    (sensor_measurements, sensor_ts) = simulate_sensors(x, y, t, add_noise=True)

    # Plotting
    plot_results(x, y, sensor_measurements, sensor_ts)

if __name__ == "__main__":
    main()
