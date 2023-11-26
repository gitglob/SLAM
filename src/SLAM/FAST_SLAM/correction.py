# Standard
# External
import numpy as np
from numpy.linalg import inv
# Local
from src.utils import normalize_angle


def measurement_model(particle, measurement):
    """
    Compute the expected measurement for a landmark and the Jacobian with respect to the landmark.

    Parameters
    ----------
    particle : Particle
        The particle representing a possible state of the robot.
    measurement : np.ndarray
        The 3x1 [id, x, y] observation of the #id landmark.

    Returns
    -------
    h : numpy.ndarray
        The expected measurement.
    H : numpy.ndarray
        The Jacobian of the measurement model.
    """
    # Extract the id and previous position of the landmark
    l_id = measurement[0]
    landmark_pos = particle.landmarks[l_id].mu

    # Calculate the expected range and bearing
    dx = landmark_pos[0] - particle.pose[0]
    dy = landmark_pos[1] - particle.pose[1]
    expected_range = np.sqrt(dx**2 + dy**2)
    expected_bearing = normalize_angle(np.arctan2(dy, dx) - particle.pose[2])
    h = np.vstack([expected_range, 
                   expected_bearing])

    # Compute the Jacobian matrix H of the measurement function h with respect to the landmark position
    H = np.zeros((2, 2))
    H[0, 0] = dx / expected_range # d(expected_range) / dl_x
    H[0, 1] = dy / expected_range # d(expected_range) / dl_y
    H[1, 0] = -dy / expected_range**2 # d(expected_bearing) / dl_x
    H[1, 1] = dx / expected_range**2 # d(expected_bearing) / dl_y

    return h, H

def initLandmarkPosition(robot_pose, z):
    """
    Initialize the estimated position of a new landmark based on the robot's current robot_pose and the measurement reading.
    
    This function calculates the position of a landmark by projecting the measured distance and bearing 
    from the robot's current position and orientation.

    Parameters
    ----------
    robot_pose : np.ndarray
        The pose of the robot, [x, y, theta], where
        x, y are the coordinates and theta is the orientation.
        
    z : np.ndarray
        The measurement vector for the landmark, [r, phi], where
        r is the distance to the landmark and phi is the bearing (angle) from the robot's perspective.

    Returns
    -------
    np.ndarray
        The estimated position of the landmark, as a 2x1 vector [x, y].
    """
    # Calculate landmark position based on robot's position and measurement
    lx = robot_pose[0] + z[0] * np.cos(normalize_angle(z[1] + robot_pose[2]))
    ly = robot_pose[1] + z[0] * np.sin(normalize_angle(z[1] + robot_pose[2]))

    # Stack the landmark's x and y position into a single array
    landmark_pos = np.vstack([lx, 
                              ly])

    return landmark_pos


def correct(particles, measurements, Q):
    """
    Performs the correction step on all particles using the landmark observations.
    
    Parameters
    ----------
    particles : list of Particle
        The list of particles representing the belief about the robot's position.
    z : list of Observations
        The observations made at the current timestep.
    Q : numpy.ndarray
        The sensor noise matrix.
        
    Returns
    --------
    list of Particle
        The list of updated particles after the correction step.
    """
    # Process each particle
    for particle in particles:
        # Process each measurement
        for measurement in measurements:
            # Get the id of the landmark corresponding to the j-th observation
            l_id = measurement[0]
            # Get the landmark actual measurement
            z = measurement[1:3]

            # If the landmark is observed for the first time:
            if not particle.landmarks[l_id].observed:
                # Initialize its position based on the measurement and the current robot pose
                particle.landmarks[l_id].mu = initLandmarkPosition(particle.pose, z)

                # Calculate Jacobian
                _, H = measurement_model(particle, measurement)

                # Initialize covariance
                particle.landmarks[l_id].sigma = inv(H) @ Q @ inv(H).T

                # Indicate that this landmark has been observed
                particle.landmarks[l_id].observed = True

                # Default importance weight is set on initialization
            else:
                ## EKF update
                # get the expected measurement
                expected_z, H = measurement_model(particle, measurement)

                # Compute the measurement covariance
                Q = (H @ particle.landmarks[l_id].sigma @ H.T) + Q

                # Calculate the Kalman gain
                K = particle.landmarks[l_id].sigma @ H.T @ inv(Q)

                # Compute the error between the z and expected_z
                Z_diff = np.vstack([z[0] - expected_z[0], 
                                    normalize_angle(z[1] - expected_z[1])])

                # Update the mean and covariance of the EKF for this landmark
                particle.landmarks[l_id].mu = particle.landmarks[l_id].mu + K @ (Z_diff)
                particle.landmarks[l_id].sigma = (np.eye(2) - K @ H) @ particle.landmarks[l_id].sigma

                ## Compute the likelihood of this observation, multiply with the former weight
                # to account for observing several features in one time step
                particle.weight = particle.weight * 1/np.sqrt(np.linalg.det(2 * np.pi * Q)) * np.exp(-0.5 * Z_diff.T @ inv(Q) @ Z_diff)

    return particles
