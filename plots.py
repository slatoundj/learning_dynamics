import numpy as np
from matplotlib import pyplot as plt

from utils import distrib_to_matrix


def plot_distribution(distribution):
    plt.figure("stationary distrib")
    x = np.arange(distribution.shape[0])
    plt.plot(x, distribution)
    plt.show()


def plot_distribution_grid(distribution, pop_infos):
    Zr, Zp, Z = pop_infos
    matrix = distrib_to_matrix(distribution, Zr+1, Zp+1)
    m = matrix/matrix.max()
    m = 1 - m

    plt.figure("stationary_distribution_grid")
    plt.pcolormesh(m, cmap="grey", shading="gouraud")
    plt.axis('scaled')
    plt.xlim(0, Zr)
    plt.ylim(0, Zp)
    plt.show()
    
    
def plot_gradient_selection(gradient, pop_infos):
    Zr, Zp, Z = pop_infos
    nabla_r, nabla_p = gradient
    dx = np.array(nabla_r)
    dy = np.array(nabla_p)
    
    X, Y = np.meshgrid(np.arange(Zr+1), np.arange(Zp+1))

    plt.figure("gradient selection")
    plt.quiver(X[::5], Y[::5], dx[::5], dy[::5], pivot='mid', units='inches')
    plt.xlim(0,Zr)
    plt.ylim(0,Zp)
    #plt.axis("scaled")
    plt.show()
    
    
def plot_gradient_with_distrib(gradient, distribution, pop_infos):
    Zr, Zp, Z = pop_infos
    nabla_r, nabla_p = gradient
    dx = np.array(nabla_r)
    dy = np.array(nabla_p)
    X, Y = np.meshgrid(np.arange(Zr+1), np.arange(Zp+1))
    
    matrix = distrib_to_matrix(distribution, Zr+1, Zp+1)
    m = matrix/matrix.max()
    m = 1 - m

    plt.figure("Stationary distribution and gradient of selection")
    plt.quiver(X[::5], Y[::5], dx[::5], dy[::5], pivot='mid', units='inches')
    plt.pcolormesh(m, cmap="grey", shading="gouraud")
    plt.xlim(0,Zr)
    plt.ylim(0,Zp)
    plt.axis("scaled")
    plt.show()
    
    
def plot_grp_achievement_in_function_of_risk(grp_achievement, risk):
    color = ["b", "r", "grey", "g", "purple"]
    plt.figure("Group achievement in function of risk")
    for i in range(len(grp_achievement)):
        print(risk[i])
        print(grp_achievement[i])
        plt.plot(risk[i], grp_achievement[i], color=color[i%5])
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()