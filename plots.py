import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

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
    plt.axis("scaled")
    plt.xlim(0,Zr)
    plt.ylim(0,Zp)
    plt.show()
    

def plot_gradient_with_distrib3(gradient, distribution, p_max, n_max, pop_infos, obstination=False, obstination_info=None):
    Zr, Zp, Z = pop_infos
    fig, axs = plt.subplots(1, 3)
    for i in range(3):
        ir_min = 0
        ip_min = 0
        if obstination and obstination_info[i] != None:
            frac, w_class = obstination_info[i]
            if w_class == "Rich":
                ir_min = int(frac*Zr)
            elif w_class == "Poor":
                ip_min = int(frac*Zp)
        nabla_r, nabla_p = gradient[i]
        dx = np.array(nabla_r)
        dy = np.array(nabla_p)
        X, Y = np.meshgrid(np.arange(Zr-ir_min+1) + ir_min, np.arange(Zp-ip_min+1) + ip_min)
        matrix = distrib_to_matrix(distribution[i], Zr-ir_min+1, Zp-ip_min+1)
        m = matrix/p_max[i]
        
        col = np.sqrt(1/2*dx**2 + 1/2*dy**2)
        col /= col.sum()
        col /= n_max[i]
        
        m1 = axs[i].pcolormesh(X, Y, m, cmap=mpl.colormaps.get_cmap("binary"), shading="gouraud")
        m2 = axs[i].quiver(X[::6], Y[::6], dx[::6], dy[::6], col[::6], pivot='mid', scale_units='xy', scale=0.02, headwidth=12, headlength=12)
        axs[i].axis("scaled")
        axs[i].set_xlim(0,Zr)
        axs[i].set_ylim(0,Zp)
        print(axs[i].get_ylim())
        print(axs[i].get_xlim())
        axs[i].set_xticks([0, 20, 40])
        axs[i].set_yticks([0, 20, 40, 60, 80, 100, 120, 140, 160])
        axs[i].set_xlabel("ir")
        axs[i].set_ylabel("ip")
    fig.colorbar(m1,ticks=[], label="stationary distribution (p)")
    fig.colorbar(m2,ticks=[], label="gradient of selection (∇)")
    plt.show()
    
    
def plot_gradient_with_distrib(gradient, distribution, p_max, n_max, pop_infos, obstination=False, obstination_info=None):
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
    
    col = np.sqrt(1/2*dx**2 + 1/2*dy**2)
    col /= col.sum()
    col /= n_max

    plt.figure("Stationary distribution and gradient of selection")
    plt.pcolormesh(X, Y, m, cmap=mpl.colormaps.get_cmap("binary"), shading="gouraud")
    plt.colorbar(ticks=[], label="stationary distribution (p)")
    plt.quiver(X[::6], Y[::6], dx[::6], dy[::6], col[::6], pivot='mid', scale_units='xy', scale=0.02, headwidth=12, headlength=12)
    plt.colorbar(ticks=[], label="gradient of selection (∇)")
    plt.axis('scaled')
    plt.xlim(0,Zr)
    plt.ylim(0,Zp)
    plt.xlabel("ir")
    plt.ylabel("ip")
    plt.show()
    
    
    
def plot_grp_achievement_in_function_of_risk(grp_achievement, risk):
    color = ["b", "r", "grey"]
    plt.figure("Group achievement in function of risk")
    plt.rc('text', usetex = True)
    ax = plt.axes()
    ax.plot(risk[0], grp_achievement[0], color=color[0], label="with inequality \& h = 0")
    ax.plot(risk[1], grp_achievement[1], color=color[1], label="with inequality \& h = 1")
    ax.plot(risk[2], grp_achievement[2], color=color[2], label="without inequality")
    ax.set_ylabel("group achievement ($\eta_G$)")
    ax.set_xlabel("risk (r)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="best")
    plt.show()