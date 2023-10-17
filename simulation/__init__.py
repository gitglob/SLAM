# Sensors noise
range_noise_std = 0.1 # meters
yaw_noise_std = 0.01 # rads

# Sensor frequency
range_frequency = 2 # 2/sec
yaw_frequency = 0.2 # 1/5sec

# Trajectory timesteps configuration
timesteps = 1001 # Number of timesteps
timestep_duration = 0.1 # Duration of each timestep (sec)

# Simulation parameters
initial_radius = 1e-6
growth_factor = 0.1
num_spirals = 5