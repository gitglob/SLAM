# Standard
import os
# External
import numpy as np
import matplotlib.pyplot as plt
# Local
from .utils import drawprobellipse, compute_sigma_points, recover_gaussian, transform


def main():
    # Initial distribution
    sigma = 0.1 * np.eye(2)
    mu = np.array([[1], [2]])
    n = len(mu)

    # Compute lambda
    alpha = 0.9
    beta = 2
    kappa = 1
    lambda_ = alpha ** 2 * (n + kappa) - n

    # Compute the sigma points corresponding to mu and sigma
    sigma_points, w_m, w_c = compute_sigma_points(mu, sigma, lambda_, alpha, beta)

    # Set up the plot
    plt.figure()
    plt.grid(True)

    # Plot original distribution with sampled sigma points
    plt.plot(mu[0], mu[1], 'ro', markersize=12, linewidth=3)
    plt.legend(['original distribution'])
    drawprobellipse(mu, sigma, 0.9, 'r')
    plt.plot(sigma_points[0, :], sigma_points[1, :], 'kx', markersize=10, linewidth=3)

    # Transform sigma points
    f_number = 3
    sigma_points_trans = transform(sigma_points, function_number=f_number)

    # Recover mu and sigma of the transformed distribution
    mu_trans, sigma_trans = recover_gaussian(sigma_points_trans, w_m, w_m)

    # Plot transformed sigma points with corresponding mu and sigma
    plt.plot(mu_trans[0], mu_trans[1], 'bo', markersize=12, linewidth=3)
    plt.legend(['transformed distribution'])
    drawprobellipse(mu_trans, sigma_trans, 0.9, 'b')
    plt.plot(sigma_points_trans[0, :], sigma_points_trans[1, :], 'kx', markersize=10, linewidth=3)

    # Figure axes setup
    plt.title('Unscented Transform', fontsize=20)
    x_min = min(mu[0], mu_trans[0])
    x_max = max(mu[0], mu_trans[0])
    y_min = min(mu[1], mu_trans[1])
    y_max = max(mu[1], mu_trans[1])
    plt.axis('equal')
    plt.axis([x_min-3, x_max+3, y_min-3, y_max+3])

    # Save plot
    directory = f"results/other/unscented_transform"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{directory}/unscented-f{f_number}.png"

    plt.savefig(filename, format='png')


if __name__ == "__main__":
    main()