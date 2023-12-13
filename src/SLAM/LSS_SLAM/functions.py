# Standard
# External
import numpy as np
from scipy.sparse import lil_matrix
# Local
from .utils import t2v, v2t, nnz_of_graph


def compute_global_error(g):
    """
    Computes the total error of the graph.

    Parameters
    ----------
    g : dict
        The graph data structure.

    Returns
    -------
    float
        The total error of the graph.
    """
    Fx = 0

    # Loop over all edges
    for edge in g['edges']:
        # Pose-pose constraint
        if edge['type'] == 'P':
            x1 = v2t(g['x'][edge['fromIdx']:edge['fromIdx']+3])
            x2 = v2t(g['x'][edge['toIdx']:edge['toIdx']+3])
            Z = v2t(edge['measurement'])
            omega = edge['information']

            error = np.linalg.inv(Z) @ (np.linalg.inv(x1) @ x2)
            eij = t2v(error)
            ei = eij.T @ omega @ eij

            Fx += ei

        # Pose-landmark constraint
        elif edge['type'] == 'L':
            x = g['x'][edge['fromIdx']:edge['fromIdx']+3]
            l = g['x'][edge['toIdx']:edge['toIdx']+2]

            X = v2t(x)
            Z = v2t(np.append(edge['measurement'], 0))
            L = v2t(np.append(l, 0))

            omega = edge['information']

            e_tmp = t2v(np.linalg.inv(Z) @ (np.linalg.inv(X) @ L))[:2]
            eij = e_tmp.T @ omega @ e_tmp

            Fx += eij

    return Fx

def linearize_and_solve(g):
    """
    Performs one iteration of the Gauss-Newton algorithm. Each constraint is linearized
    and added to the Hessian.

    Parameters
    ----------
    g : dict
        The graph data structure.

    Returns
    -------
    numpy.ndarray
        The change in the state vector after solving the linear system.
    """
    nnz = nnz_of_graph(g)

    # Allocate the sparse H and the vector b
    H = lil_matrix((len(g['x']), len(g['x'])))
    b = np.zeros(len(g['x']))

    need_to_add_prior = True

    # Compute the addend term to H and b for each of our constraints
    print('Linearizing and building system')
    for edge in g['edges']:
        if edge['type'] == 'P':
            x1 = g['x'][edge['fromIdx']:edge['fromIdx']+3]
            x2 = g['x'][edge['toIdx']:edge['toIdx']+3]

            # Compute the error and the Jacobians
            e, A, B = linearize_pose_pose_constraint(x1, x2, edge['measurement'])

            i = slice(edge['fromIdx'], edge['fromIdx']+3)
            j = slice(edge['toIdx'], edge['toIdx']+3)

            omega = edge['information']

            # Update b and H
            b[i] += A.T @ omega @ e
            b[j] += B.T @ omega @ e

            H[i, i] += A.T @ omega @ A
            H[i, j] += A.T @ omega @ B
            H[j, i] += B.T @ omega @ A
            H[j, j] += B.T @ omega @ B

            if need_to_add_prior:
                H[0:3, 0:3] += np.eye(3)
                need_to_add_prior = False

        elif edge['type'] == 'L':
            x1 = g['x'][edge['fromIdx']:edge['fromIdx']+3]
            x2 = g['x'][edge['toIdx']:edge['toIdx']+2]

            # Compute the error and the Jacobians
            e, A, B = linearize_pose_landmark_constraint(x1, x2, edge['measurement'])

            i = slice(edge['fromIdx'], edge['fromIdx']+3)
            j = slice(edge['toIdx'], edge['toIdx']+2)

            omega = edge['information']

            # Update b and H
            b[i] += A.T @ omega @ e
            b[j] += B.T @ omega @ e

            H[i, i] += A.T @ omega @ A
            H[i, j] += A.T @ omega @ B
            H[j, i] += B.T @ omega @ A
            H[j, j] += B.T @ omega @ B

    print('Solving system')
    # Solve the linear system
    dx = -np.linalg.solve(H.tocsr(), b)

    return dx

def linearize_pose_landmark_constraint(x, l, z):
    """
    Compute the error and Jacobians of a pose-landmark constraint.

    Parameters
    ----------
    x : numpy.ndarray
        3x1 vector (x, y, theta) of the robot pose.
    l : numpy.ndarray
        2x1 vector (x, y) of the landmark.
    z : numpy.ndarray
        2x1 vector (x, y) of the measurement, the position of the landmark in
        the coordinate frame of the robot given by the vector x.

    Returns
    -------
    e : numpy.ndarray
        2x1 error of the constraint.
    A : numpy.ndarray
        2x3 Jacobian with respect to x.
    B : numpy.ndarray
        2x2 Jacobian with respect to l.
    """
    X = v2t(x)
    Ri = X[0:2, 0:2]

    e = Ri.T @ (l - x[0:2]) - z

    theta_i = np.arctan2(Ri[1, 0], Ri[0, 0])
    xi, yi = x[0], x[1]
    xl, yl = l[0], l[1]

    eij_xi = np.array([-np.cos(theta_i), np.sin(theta_i)])
    eij_yi = np.array([-np.sin(theta_i), -np.cos(theta_i)])
    eij_theta_i = np.array([-(xl-xi)*np.sin(theta_i) + (yl-yi)*np.cos(theta_i),
                            -(xl-xi)*np.cos(theta_i) - (yl-yi)*np.sin(theta_i)])

    A = np.column_stack([eij_xi, eij_yi, eij_theta_i])

    eij_xl = np.array([np.cos(theta_i), -np.sin(theta_i)])
    eij_yl = np.array([np.sin(theta_i), np.cos(theta_i)])

    B = np.column_stack([eij_xl, eij_yl])

    return e, A, B

def linearize_pose_pose_constraint(x1, x2, z):
    """
    Compute the error and Jacobians of a pose-pose constraint.

    Parameters
    ----------
    x1 : numpy.ndarray
        3x1 vector (x, y, theta) of the first robot pose.
    x2 : numpy.ndarray
        3x1 vector (x, y, theta) of the second robot pose.
    z : numpy.ndarray
        3x1 vector (x, y, theta) of the measurement.

    Returns
    -------
    e : numpy.ndarray
        3x1 error of the constraint.
    A : numpy.ndarray
        3x3 Jacobian with respect to x1.
    B : numpy.ndarray
        3x3 Jacobian with respect to x2.
    """
    X1 = v2t(x1)
    X2 = v2t(x2)
    Z = v2t(z)

    e = t2v(np.linalg.inv(Z) @ (np.linalg.inv(X1) @ X2))

    Rij = Z[0:2, 0:2]
    Ri = X1[0:2, 0:2]

    theta_i = np.arctan2(Ri[1, 0], Ri[0, 0])
    theta_ij = np.arctan2(Rij[1, 0], Rij[0, 0])

    xi, yi = x1[0], x1[1]
    xj, yj = x2[0], x2[1]

    eij_xi = np.array([-np.cos(theta_i)*np.cos(theta_ij) + np.sin(theta_i)*np.sin(theta_ij),
                       -np.sin(theta_i)*np.cos(theta_ij) - np.cos(theta_i)*np.sin(theta_ij),
                        0])
    eij_yi = np.array([np.sin(theta_i)*np.cos(theta_ij) + np.cos(theta_i)*np.sin(theta_ij),
                       -np.sin(theta_i)*np.sin(theta_ij) + np.cos(theta_i)*np.cos(theta_ij),
                        0])
    eij_theta_i = np.array([-(xj - xi)*(np.sin(theta_i)*np.cos(theta_ij) + np.cos(theta_i)*np.sin(theta_ij)) + 
                             (yj - yi)*(np.cos(theta_i)*np.cos(theta_ij) - np.sin(theta_i)*np.sin(theta_ij)),
                             (xj - xi)*(np.sin(theta_i)*np.sin(theta_ij) - np.cos(theta_i)*np.cos(theta_ij)) - 
                             (yj - yi)*(np.cos(theta_i)*np.sin(theta_ij) + np.sin(theta_i)*np.cos(theta_ij)),
                             -1])

    A = np.column_stack([eij_xi, eij_yi, eij_theta_i])

    eij_xj = np.array([np.cos(theta_i)*np.cos(theta_ij) - np.sin(theta_i)*np.sin(theta_ij),
                       np.sin(theta_i)*np.cos(theta_ij) + np.cos(theta_i)*np.sin(theta_ij),
                       0])
    eij_yj = np.array([-np.cos(theta_i)*np.sin(theta_ij) - np.sin(theta_i)*np.cos(theta_ij),
                       np.sin(theta_i)*np.sin(theta_ij) - np.cos(theta_i)*np.cos(theta_ij),
                       0])
    eij_theta_j = np.array([0, 0, 1])

    B = np.column_stack([eij_xj, eij_yj, eij_theta_j])

    return e, A, B

