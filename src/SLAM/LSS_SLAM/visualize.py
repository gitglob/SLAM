# Standard
import os
# External
import matplotlib.pyplot as plt
# Local
from .utils import get_poses_landmarks


def plot_graph(data, iteration, dataset):
    """
    Plots a 2D SLAM graph.

    Parameters
    ----------
    data : dict
        The graph containing SLAM data.
    iteration : int, optional
        The iteration number for saving the plot, by default -1 (don't save).
    """
    plt.clf()

    p, l = get_poses_landmarks(data)

    if len(l) > 0:
        landmarkIdxX = [idx for idx in l if idx % 3 == 0]
        landmarkIdxY = [idx + 1 for idx in landmarkIdxX]
        x = [data['x'][idx].item() for idx in landmarkIdxX]
        y = [data['x'][idx].item() for idx in landmarkIdxY]
        plt.scatter(x, y, marker='o', facecolors='none', edgecolors='red', s=40)

    if len(p) > 0:
        pIdxX = [idx for idx in p if idx % 3 == 0]
        pIdxY = [idx + 1 for idx in pIdxX]
        x = [data['x'][idx].item() for idx in pIdxX]
        y = [data['x'][idx].item() for idx in pIdxY]
        plt.scatter(x, y, marker='x', c='blue', s=40)

    # Draw line segments (if needed)
    # Code for drawing line segments can be added here, similar to the above plots

    directory = f'results/slam/LS/{dataset}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'{directory}/step_{iteration:03d}.png'
    plt.savefig(filename)

    plt.close()

