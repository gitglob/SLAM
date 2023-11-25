# Standard
# External
import numpy as np
# Local


def correct(particles, z, Q_t):
    """
    Performs the correction step on all particles using the landmark observations.
    
    Parameters
    ----------
    particles : list of Particle
        The list of particles representing the belief about the robot's position.
    z : list of Observations
        The observations made at the current timestep.
    Q_t : numpy.ndarray
        The sensor noise matrix.
        
    Returns
    --------
    list of Particle
        The list of updated particles after the correction step.
    """
    num_particles = len(particles)
    m = len(z)  # Number of measurements in this time step

    # Process each particle
    for particle in particles:
        robot = particle.pose

        # Process each measurement
        for j in range(m):
            # Get the id of the landmark corresponding to the j-th observation
            l = z[j].id

            # If the landmark is observed for the first time:
            if not particle.landmarks[l].observed:
                # Initialize its position based on the measurement and the current robot pose
                # ... (initialization code goes here)
                # Indicate that this landmark has been observed
                particle.landmarks[l].observed = True

            else:
                # get the expected measurement
                expected_z, H = measurement_model(particle, z[j])

                # Compute the measurement covariance
                # ... (computation code goes here)

                # Calculate the Kalman gain
                # ... (Kalman gain code goes here)

                # Compute the error between the z and expectedZ (normalize the angle)
                # ... (error computation code goes here)

                # Update the mean and covariance of the EKF for this landmark
                # ... (update code goes here)

                # Compute the likelihood of this observation, multiply with the former weight
                # to account for observing several features in one time step
                # ... (likelihood computation code goes here)

    return particles
