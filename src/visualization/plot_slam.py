# Standard
# External
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from scipy.stats import chi2
# Local

def normalize_angle(phi):
    """Normalize phi to be between -pi and pi."""
    while phi > np.pi:
        phi -= 2 * np.pi
    while phi < -np.pi:
        phi += 2 * np.pi
    return phi

def normalize_all_bearings(z):
    """Go over the observations vector and normalize the bearings.
    The expected format of z is [range, bearing, range, bearing, ...]"""
    
    for i in range(1, len(z), 2):
        z[i] = normalize_angle(z[i])
    return z

def drawellipse(x, a, b, color):
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
    plt.show()

def drawprobellipse(x, C, alpha, color):    
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
    Draw a rectangle in matplotlib.

    Parameters:
    - pose: A list or tuple containing the pose of the rectangle [x, y, theta].
    - length: The length of the rectangle.
    - width: The width of the rectangle.
    - corner_radius: Radius of the rounded corners.
    - fill_flag: 1 to fill the rectangle with the specified color, 0 otherwise.
    - color: Color of the rectangle.

    Returns:
    - Patch object representing the rectangle.
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

def plot_state(mu, state_cov, landmarks, timestep, observedLandmarks, z):
    """
    Visualizes the state of the EKF SLAM algorithm.

    The resulting plot displays the following information:
    - mu: Current robot state estimate
    - state_cov: Current robot state uncertainty
    - landmarks: Map landmarks ground truth (black)
    - current robot pose estimate (red)
    - current landmark pose estimates (blue)
    - visualization of the observations made at this time step (line between robot and landmark)
    """

    plt.figure()
    plt.clf()
    plt.grid(True)
    
    # Convert the landmarks struct to a more Pythonic structure
    L = np.vstack([landmarks["xy"]])

    drawprobellipse(mu[:3], state_cov[:3,:3], 0.6, 'r')
    plt.plot(L[:,0], L[:,1], 'k+', markersize=10, linewidth=5)

    for i, observed in enumerate(observedLandmarks):
        if observed:
            plt.plot(mu[2*i + 3], mu[2*i + 4], 'bo', markersize=10, linewidth=5)
            drawprobellipse(mu[2*i + 3 : 2*i + 5], state_cov[2*i + 3 : 2*i + 5, 2*i + 3 : 2*i + 5], 0.6, 'b')
            
    for obs in z:
        id = obs[0]
        mX = mu[2*id + 3]
        mY = mu[2*id + 4]
        plt.plot([mu[0], mX], [mu[1], mY], 'k', linewidth=1)
        
    drawrobot(mu[:3], 'r', 3, 0.3, 0.3)
    plt.xlim([-2, 12])
    plt.ylim([-2, 12])

    filename = f'results/slam/ekf_{timestep:03}.png'
    plt.savefig(filename)

    plt.close()
