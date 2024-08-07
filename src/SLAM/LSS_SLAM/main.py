# Standard
# External
import numpy as np
# Local
from .data import load_data
from .functions import compute_global_error, linearize_and_solve
from .visualize import plot_graph


def main():
    # Load the graph
    datasets = ["simulation-pose-landmark",
                "simulation-pose-pose",
                "intel",
                "dlr"]
    dataset = datasets[3]
    datapath = "exercises/16_lsslam_framework/data/" + dataset + ".dat"
    x, edges, idLookup = load_data(datapath)
    data = {
        "x": x,
        "edges": edges,
        "idLookup": idLookup        
    }

    # Plot the initial state of the graph
    plot_graph(data, 0, dataset)

    print(f'Initial error {compute_global_error(data)}')

    # Number of iterations
    num_iterations = 100

    # Maximum allowed dx
    EPSILON = 1e-4

    # Carry out the iterations
    for i in range(num_iterations):
        print(f'Performing iteration {i}')

        dx = linearize_and_solve(data)

        # Apply the solution to the state vector g.x
        data['x'] = data['x'] + dx

        # Plot the current state of the graph
        plot_graph(data, i, dataset)

        err = compute_global_error(data)

        # Print current error
        print(f'Current error {err}')

        # Termination criterion
        if np.max(np.abs(dx)) < EPSILON:
            break

    print(f'Final error {err}')

if __name__ == "__main__":
    main()
    