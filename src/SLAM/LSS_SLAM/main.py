# Standard
import os
import time
# External
import numpy as np
from scipy.io import loadmat
# Local
from .data import load_data
from .functions import compute_global_error, linearize_and_solve
from .visualize import plot_graph


def main():
    # Assuming plot_graph, compute_global_error, linearize_and_solve, and other necessary functions are defined

    # Load the graph
    dataset = "simulation-pose-landmark"
    datapath = "exercises/16_lsslam_framework/data/" + dataset + ".dat"
    data = loadmat(datapath)
    g = data['g']

    # Plot the initial state of the graph
    plot_graph(g, 0, dataset)

    print(f'Initial error {compute_global_error(g)}')

    # Number of iterations
    num_iterations = 100

    # Maximum allowed dx
    EPSILON = 1e-4

    # Carry out the iterations
    for i in range(1, num_iterations + 1):
        print(f'Performing iteration {i}')

        dx = linearize_and_solve(g)

        # Apply the solution to the state vector g.x
        g['x'] = g['x'] + dx

        # Plot the current state of the graph
        plot_graph(g, i, dataset)

        err = compute_global_error(g)

        # Print current error
        print(f'Current error {err}')

        # Termination criterion
        if np.max(np.abs(dx)) < EPSILON:
            break

    print(f'Final error {err}')

if __name__ == "__main__":
    main()
    