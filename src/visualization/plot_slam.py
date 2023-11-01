# Standard
# External
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from scipy.stats import chi2
# Local

def drawellipse(x, a, b, color):
    """
    Draw an ellipse with specified parameters on a 2D plane.

    Parameters
    ----------
    x : tuple
        A tuple containing the center (xo, yo) of the ellipse and rotation angle in radians. Format: (xo, yo, angle).
    a : float
        Semi-major axis length of the ellipse.
    b : float
        Semi-minor axis length of the ellipse.
    color : str
        Color of the ellipse to be drawn.

    Notes
    -----
    - This function assumes that necessary libraries like numpy (as np) and matplotlib.pyplot (as plt) are already imported.
    - The ellipse is plotted with a resolution defined by NPOINTS.
    """
    
    NPOINTS = 100  # point density or resolution

    # Compose point vector
    ivec = np.linspace(0, 2*np.pi, NPOINTS)  # index vector
    p = np.zeros((2, NPOINTS))
    p[0, :] = a * np.cos(ivec)
    p[1, :] = b * np.sin(ivec)

    # Translate and rotate
    xo, yo, angle = x
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]]).reshape((2,2))
    T = np.vstack([xo, yo]) @ np.ones((1, NPOINTS))
    p = R @ p + T

    # Plot
    plt.plot(p[0, :], p[1, :], color=color, linewidth=2)
    plt.axis('equal')  # to make sure the ellipse looks correct

def drawprobellipse(x, C, alpha, color):
    """
    Draw a probability ellipse based on covariance matrix and confidence level.

    Parameters
    ----------
    x : np.ndarray
        A 2-element array containing the x and y coordinates of the ellipse center.
    C : np.ndarray
        2x2 covariance matrix.
    alpha : float
        Confidence level for which the ellipse will represent.
        Should be in the interval (0, 1). E.g., 0.95 for a 95% confidence ellipse.
    color : str
        Color of the ellipse to be drawn.

    Notes
    -----
    - The function requires numpy (as np) and assumes the `drawellipse` function is defined and available.
    - The ellipse is calculated based on the provided covariance matrix and scaled according to the desired confidence level.
    """

    # Calculate unscaled half axes
    sxx = C[0,0]
    syy = C[1,1]
    sxy = C[0,1]
    a = np.sqrt(0.5 * (sxx + syy + np.sqrt((sxx - syy)**2 + 4 * sxy**2)))  # always greater
    b = np.sqrt(0.5 * (sxx + syy - np.sqrt((sxx - syy)**2 + 4 * sxy**2)))  # always smaller

    # Remove imaginary parts in case of neg. definite C
    a = np.real(a)
    b = np.real(b)

    # Scaling in order to reflect specified probability
    a = a * np.sqrt(chi2.ppf(alpha, 2))
    b = b * np.sqrt(chi2.ppf(alpha, 2))

    # Look where the greater half axis belongs to
    if sxx < syy:
        a, b = b, a

    # Calculate inclination (numerically stable)
    if sxx != syy:
        angle = 0.5 * np.arctan(2 * sxy / (sxx - syy))
    elif sxy == 0:
        angle = 0  # angle doesn't matter 
    elif sxy > 0:
        angle = np.pi / 4
    elif sxy < 0:
        angle = -np.pi / 4
        
    x = [x[0].item(), x[1].item(), angle]

    # Use the drawellipse function
    drawellipse(x, a, b, color)

def drawrect(pose, length, width, corner_radius, fill_flag, color):
    """
    Draw a rotated rectangle (with optionally rounded corners) on a 2D plane.

    Parameters
    ----------
    pose : tuple
        A tuple containing the x and y coordinates of the rectangle's center, and its rotation angle in radians.
        Format: (x, y, theta).
    length : float
        Length (or major axis) of the rectangle.
    width : float
        Width (or minor axis) of the rectangle.
    corner_radius : float
        Radius of the rounded corners. Use 0 for sharp corners.
    fill_flag : bool
        True if the rectangle should be filled, False otherwise.
    color : str
        Color of the rectangle to be drawn.

    Returns
    -------
    rectangle : matplotlib.patches.FancyBboxPatch
        The drawn rectangle object.

    Notes
    -----
    - This function assumes that necessary libraries like numpy (as np) and matplotlib.pyplot (as plt) and
      matplotlib.patches (as mpatches) are already imported.
    """
    x, y, theta = pose
    rectangle = mpatches.FancyBboxPatch((x - length/2, y - width/2), 
                                        length, width, 
                                        boxstyle=f"round,pad=0,rounding_size={corner_radius}", 
                                        fill=fill_flag, 
                                        color=color)

    t = plt.gca().transData
    t_rotate = t + plt.Affine2D().rotate_deg_around(x, y, np.degrees(theta))
    rectangle.set_transform(t_rotate)

    plt.gca().add_patch(rectangle)

    return rectangle

def drawrobot(xvec, color, robot_type=2, B=0.4, L=0.6):
    """
    Draw a representation of the robot on the current plot.

    Parameters
    ----------
    xvec : np.ndarray or list
        A vector of shape (3,) representing the robot's [x, y, theta] state.
    color : str
        The color to use for the robot representation.
    robot_type : int, optional
        The type of robot representation to draw. Supported types are:
        0 - Origin cross
        1 - Wheel pair with axis and arrow
        2 - Wheel pair with axis, arrow, and circular contour
        3 - Circular contour with centerline
        4 - Wheel pair with axis, arrow, and rectangular contour
        5 - Rectangular contour with centerline
        Default is 2.
    B : float, optional
        Robot's wheelbase width or relevant dimension depending on robot type. Default is 0.4.
    L : float, optional
        Robot's length or relevant dimension depending on robot type. Default is 0.6.

    Notes
    -----
    - The function assumes the `drawrect` and `drawellipse` functions are defined and available.
    - Robot's representation is drawn on the current active plot.
    """

    x, y, theta = xvec
    T = np.array([x, y]).reshape((2,1))
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).reshape((2,2))
    
    # Constants
    CS = 0.1
    WD = 0.2
    WT = 0.03
    RR = WT / 2
    RRR = 0.04
    HL = 0.09
    
    if robot_type == 0:
        # Draw origin cross
        p = R @ np.array([[CS, -CS, 0, 0], [0, 0, -CS, CS]]) + T.reshape(-1, 1)
        plt.plot(p[0, :2], p[1, :2], color=color)
        plt.plot(p[0, 2:], p[1, 2:], color=color)

    elif robot_type == 1 or robot_type == 2 or robot_type == 4:
        # Draw wheel pair with axis and arrow
        xlw = np.array([x + B/2 * np.cos(theta + np.pi/2), y + B/2 * np.sin(theta + np.pi/2), theta])
        drawrect(xlw, WD, WT, RR, 1, color)
        
        xlw = np.array([x - B/2 * np.cos(theta + np.pi/2), y - B/2 * np.sin(theta + np.pi/2), theta])
        drawrect(xlw, WD, WT, RR, 1, color)
        
        # Draw axis cross with arrow
        p = R @ np.array([[0, 0], [-B/2 + WT/2, B/2 - WT/2]]) + T.reshape(-1, 1)
        plt.plot(p[0, :], p[1, :], color=color)
        
        if robot_type == 2:
            # Draw circular contour for type 2
            drawellipse(xvec, (B + WT)/2, (B + WT)/2, color)
            
        elif robot_type == 4:
            # Draw rectangular contour for type 4
            drawrect(xvec, L, B, RRR, 0, color)

    elif robot_type == 3:
        # Draw circular contour
        drawellipse(xvec, (B + WT)/2, (B + WT)/2, color)
        p = R @ np.array([(B + WT)/2, 0]).reshape((2,1)) + T
        plt.plot([T[0], p[0]], [T[1], p[1]], color=color, linewidth=2)

    elif robot_type == 5:
        # Draw rectangular contour
        drawrect(xvec, L, B, RRR, 0, color)
        p = R @ np.array([L/2, 0]).reshape((2,1)) + T
        plt.plot([T[0], p[0]], [T[1], p[1]], color=color, linewidth=2)

    else:
        print("Unsupported robot type")

def plot_state(mu, state_cov, timestep, landmarks, all_seen_landmarks, observed_landmarks):
    """
    Visualizes the state of the EKF SLAM algorithm.

    Parameters
    ----------
    mu : np.ndarray
        Current estimate of the robot's state and landmarks, 
        starting with the robot's x, y, and theta, followed by x, y of each landmark.
    state_cov : np.ndarray
        Current state covariance matrix. Represents the uncertainty in the robot's state and landmark estimates.
    landmarks : dict
        Dictionary containing the ground truth positions of landmarks in the map.
    timestep : int
        The current time step for which this state represents.
    all_seen_landmarks : list of bool
        List indicating which landmarks have been observed up to the current time step.
    observed_landmarks : list of tuples
        List of observations made at the current time step. Each tuple contains (landmark ID, observation data).

    Notes
    -----
    - The resulting plot displays the following information:
      - `mu`: Current robot state estimate (red).
      - `state_cov`: Uncertainty ellipses around the robot's state and landmark estimates.
      - `landmarks`: Ground truth landmark positions (black `+` markers).
      - Estimated landmark positions (blue `o` markers).
      - Observations made at the current time step (lines between robot and observed landmarks).
    - The function assumes `drawprobellipse` and `drawrobot` functions are defined and available.
    - The plot is saved as a PNG file in the 'results/slam/' directory.
    """

    plt.figure()
    plt.clf()
    plt.grid(True)

    # Draw the robot state
    drawrobot(mu[:3], 'r', 3, 0.3, 0.3)

    # Draw a probability ellipse based on the covariance matrix of the robot
    drawprobellipse(mu[:3], state_cov[:3,:3], 0.6, 'r')

    # Draw the ground truth of the landmarks
    for i in range(len(landmarks["xy"])):
        plt.scatter(landmarks["xy"][i][0], landmarks["xy"][i][1], c='k', marker='+', s=100) # label=f"Landmark #{landmarks['ids'][i]} - G.T."

    for l in all_seen_landmarks:
        # Draw the curently observed landmarks
        plt.scatter(mu[2*l + 3], mu[2*l + 4], c='b', marker='o', s=10) # , label=f"Landmark #{l} - state"

        # Draw a probability ellipse based on the covariance matrix of the observed landmarks
        drawprobellipse(mu[2*l + 3 : 2*l + 5], state_cov[2*l + 3 : 2*l + 5, 2*l + 3 : 2*l + 5], 0.6, 'b')
            
    # Draw the observation that the robot just made
    for obs in observed_landmarks:
        l = obs[0]
        mX = mu[2*l + 3]
        mY = mu[2*l + 4]
        plt.plot(mu[0], mX, '--k', linewidth=1) # label=f"Landmark #{l} - observation X"
        plt.plot(mu[1], mY, '--k', linewidth=1) # label=f"Landmark #{l} - observation Y"

    # plt.xlim([-2, 12])
    # plt.ylim([-2, 12])

    # plt.legend()

    filename = f'results/slam/ekf_{timestep:03}.png'
    plt.savefig(filename)

    plt.close()
