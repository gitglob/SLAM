# Standard
import os
# External
import numpy as np
import matplotlib.pyplot as plt
# Local
from . import range_noise_std, yaw_noise_std, range_frequency, yaw_frequency
from . import timesteps, timestep_duration, initial_radius, growth_factor, num_spirals


def simulate_spiral_movement(timesteps, timestep_duration, initial_radius, growth_factor, num_spirals):
    """
    Simulates a spiral movement for a vehicle.

    Parameters
    ----------
    timesteps : int
        Number of timesteps to simulate.
    timestep_duration : int
        Duration of each timestep.
    initial_radius : float
        Initial radius of the spiral.
    growth_factor : float
        Factor determining the growth of the spiral.

    Returns
    -------
    x, y, t : np.ndarray, np.ndarray, np.ndarray
        - X and Y coordinates of the vehicle over time (m).
        - Time corresponding to the coordinates (sec).
    """
    # Create time array
    t = np.linspace(0, timesteps*timestep_duration, timesteps)

    # Simulate spiral
    theta = np.linspace(0, num_spirals * 2 * np.pi, timesteps)
    r = initial_radius + growth_factor * t
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y, t

def simulate_sensors(x, y, t, range_noise_std, yaw_noise_std, range_frequency, yaw_frequency):
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
    range_t = np.arange(t[0], t[-1], 1/range_frequency)
    range_indices = [np.argmin(np.abs(t - time)) for time in range_t]
    yaw_t = np.arange(t[0], t[-1], 1/yaw_frequency)
    yaw_indices = [np.argmin(np.abs(t - time)) for time in yaw_t]
    
    # Extract range and yaw values at the specified timesteps and add measurement noise
    range_measurements = np.sqrt(x[range_indices]**2 + y[range_indices]**2) 
    range_measurements += np.random.normal(0, range_noise_std, len(range_indices))
    
    yaw_measurements = np.arctan2(y[yaw_indices], x[yaw_indices])
    yaw_measurements += np.random.normal(0, yaw_noise_std, len(yaw_indices))
    
    return (range_measurements, range_t), (yaw_measurements, yaw_t)

def polar2xy(ranges, range_ts, yaws, yaw_ts):
    """
    Convert range and yaw data to x, y coordinates using the provided timestamps.
    If no yaw measurement is available for a range timestamp, the last available yaw measurement is used.
    Also, returns a boolean mask indicating if the corresponding x, y point used the last available yaw measurement.

    Parameters
    ----------
    ranges, range_ts, yaws, yaw_ts : np.ndarray, np.ndarray
        Range and yaw measurements with their respective timestamps.

    Returns
    -------
    x, y, mask : np.ndarray, np.ndarray, np.ndarray
        Converted X and Y coordinates and a mask indicating which points used the last yaw measurement.
    """
    # Lists to store converted coordinates and mask
    x_coords = []
    y_coords = []
    mask = []

    # Initialize last available yaw value
    last_yaw = yaws[0]

    for rt in range_ts:
        # Find closest yaw timestamp
        idx = np.searchsorted(yaw_ts, rt, side='right') - 1
        if idx >= 0:
            last_yaw = yaws[idx]
            mask.append(True if yaw_ts[idx] == rt else False)
        else:
            mask.append(False)
        # Convert range and yaw to x, y
        r = ranges[np.where(range_ts == rt)]
        x_coords.append(r * np.cos(last_yaw))
        y_coords.append(r * np.sin(last_yaw))

    return np.array(x_coords), np.array(y_coords), np.array(mask)

def plot_results(x, y, ranges, range_ts, yaws, yaw_ts):
    """
    Plots the spiral movement of the robot and its sensor readings.

    Parameters
    ----------
    x, y : np.ndarray, np.ndarray
        X and Y coordinates of the vehicle.
    ranges, range_ts, yaws, yaw_ts : np.ndarray, np.ndarray
        Range and yaw measurements with their respective timestamps.

    Returns
    -------
    None
    """
    # Convert range and yaw data back to x and y
    sensor_x, sensor_y, mask = polar2xy(ranges, range_ts, yaws, yaw_ts)

    fig, ax = plt.subplots(3, 1, figsize=(8, 15))
    
    # Spiral movement plot
    ax[0].plot(x, y, label='Ground Truth Path', color='blue')
    ax[0].scatter(sensor_x[mask], sensor_y[mask], c='red', s=15, label='Sensor Readings (New Yaw)', marker='x')
    ax[0].scatter(sensor_x[~mask], sensor_y[~mask], c='purple', s=5, label='Sensor Readings (Last Yaw)', marker='o')
    ax[0].set_title('Spiral Movement of the Robot')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].grid(True)
    ax[0].axis('equal')
    ax[0].legend()

    # Range sensor readings plot
    ax[1].plot(range_ts, ranges, label='Range Readings', color='green')
    ax[1].set_title('Range Readings over Time')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Range (m)')
    ax[1].grid(True)
    ax[1].legend()

    # Yaw sensor readings plot
    ax[2].plot(yaw_ts, yaws, label='Yaw Readings', color='orange')
    ax[2].set_title('Yaw Readings over Time')
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Yaw (rad)')
    ax[2].grid(True)
    ax[2].legend()

    plt.tight_layout()
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig('results/simulated_trajectory.png')


def main():
    # Simulate spiral movement
    x, y, t = simulate_spiral_movement(timesteps, timestep_duration, initial_radius, growth_factor, num_spirals)

    # Simulate sensors
    (ranges, range_ts), (yaws, yaw_ts) = simulate_sensors(x, y, t, range_noise_std, yaw_noise_std, range_frequency, yaw_frequency)

    # Plotting
    plot_results(x, y, ranges, range_ts, yaws, yaw_ts)

if __name__ == "__main__":
    main()
