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
    
    
def plot_gradient_with_distrib(gradient, distribution, p_max, pop_infos, obstination=False, obstination_info=None):
    Zr, Zp, Z = pop_infos
    ir_min = 0
    ip_min = 0
    if obstination and obstination_info != None:
        frac, w_class = obstination_info
        if w_class == "Rich":
            ir_min = int(frac*Zr)
        elif w_class == "Poor":
            ip_min = int(frac*Zp)
    nabla_r, nabla_p = gradient
    dx = np.array(nabla_r)
    dy = np.array(nabla_p)
    X, Y = np.meshgrid(np.arange(Zr-ir_min+1) + ir_min, np.arange(Zp-ip_min+1) + ip_min)
    
    matrix = distrib_to_matrix(distribution, Zr-ir_min+1, Zp-ip_min+1)
    m = matrix/p_max
    m = 1 - m

    plt.figure("Stationary distribution and gradient of selection")
    plt.pcolormesh(X, Y, m, cmap="grey", shading="gouraud")
    plt.quiver(X[::5], Y[::5], dx[::5], dy[::5], pivot='mid', units='inches')
    plt.xlim(0,Zr)
    plt.ylim(0,Zp)
    #plt.axis("scaled")
    plt.show()
    
    
def plot_grp_achievement_in_function_of_risk(grp_achievement, risk):
    color = ["b", "r", "grey"]
    plt.figure("Group achievement in function of risk")
    plt.rc('text', usetex = True)
    ax = plt.axes()
    #for i in range(len(grp_achievement)):
    ax.plot(risk[0], grp_achievement[0], color=color[0], label="with inequality \& h = 0")
    ax.plot(risk[1], grp_achievement[1], color=color[1], label="with inequality \& h = 1")
    ax.plot(risk[2], grp_achievement[2], color=color[2], label="without inequality")
    ax.set_ylabel("group achievement ($\eta_G$)")
    ax.set_xlabel("risk (r)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="best")
    plt.show()