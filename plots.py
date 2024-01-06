import numpy as np
from matplotlib import pyplot as plt

from utils import rec_V


def plot_distribution(distribution):
    plt.figure("stationary distrib")
    x = np.arange(distribution.shape[0])
    plt.plot(x, distribution)
    plt.show()


def plot_distribution_grid(distribution, pop_infos):
    Zr, Zp, Z = pop_infos
    numStates = (Zr+1) * (Zp+1)
    matrix = np.zeros((Zp+1, Zr+1))
    for i in range(numStates):
        ir, ip = rec_V(i, Zr)
        matrix[ip, ir] = distribution[i]

    m = matrix/matrix.max()
    m = 1 - m

    plt.figure("stationary_distribution_grid")
    plt.pcolormesh(m, cmap="grey", shading="gouraud")
    plt.axis('scaled')
    plt.xlim(0, Zr)
    plt.ylim(0, Zp)
    plt.show()