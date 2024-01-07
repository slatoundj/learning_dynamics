import networkx as nx
import random
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from probabilities import payoffs
from utils import fermi
from group_achievement import gr_achievement


Zr = 40         # number of richs in the population
Zp = 160        # number of poors in the population
Z = Zr + Zp     # population size

h = 0           # homophily

# initial endowment
br = 2.5    
bp = 0.625

# average endowment
avg_b = 0.2*br + 0.8*bp

# cooperation cost
c = 0.1
cr = c * br
cp = c * bp

N = 6           # sub group size
M = 3           # 
t = M*c*avg_b   # threshold of group achievement

strategies = ["Defect", "Cooperate"]
w_classes = ["Rich", "Poor"]

mu = 1/Z        # mutation probability
beta = 3        # intensity of selection

r = 0.4         # risk perception


def transform_graph_to_dict_of_lists(graph: nx.Graph):
    """Transform a graph to a dictionnary of lists

    Args:
        graph (nx.Graph): the graph to be transformed

    Returns:
        dict[int, list[int]]: the graph in the form of a dictionary of lists
    """
    graph_dictionary = {}
    for node in graph.nodes:
        graph_dictionary[node] = list(graph[node].keys())

    return graph_dictionary


def avg_achievement(network, Z, nodes, br, bp):
    eta_g = 0
    for i in range(Z):
        neighbours = network[i]
        jr = 0
        jp = 0
        if nodes[i]["strategy"] == "Cooperate":
            if nodes[i]["w_class"] == "Rich":
                jr += 1
            else:
                jp += 1
        for j in neighbours:
            if nodes[j]["strategy"] == "Cooperate":
                if nodes[j]["w_class"] == "Rich":
                    jr += 1
                else:
                    jp += 1
        t = (0.2*br + 0.8*bp) * c * ((len(neighbours) +1))/2
        eta_g += gr_achievement((jr, jp), (cr, cp, t))
    eta_g /= Z
    return eta_g


def estimate_fitness(selected:list, network, nodes:list, r, payoffs_infos):
    """Estimate the fitness of the selected players.
    The estimated fitness is the average expected payoff of the player against its neighbours

    Args:
        selected (tuple(int, int)): the selected players
        network (dict[int, list[int]]): the network
        nodes (list[int]): the list of nodes strategies
        Z (int): the population size
        A (list[list[float]]): the expected payoffs matrix

    Returns:
        list[float]: the estimated fitness of the selected players
    """
    br, bp, cr, cp = payoffs_infos
    fitness = []
    for i in selected:
        neighbours = network[i]
        jr = 0
        jp = 0
        if nodes[i]["strategy"] == "Cooperate":
            if nodes[i]["w_class"] == "Rich":
                jr += 1
            else:
                jp += 1
        for j in neighbours:
            if nodes[j]["strategy"] == "Cooperate":
                if nodes[j]["w_class"] == "Rich":
                    jr += 1
                else:
                    jp += 1
        t = (0.2*br + 0.8*bp) * c * ((len(neighbours) +1))/2
        f = payoffs((jr, jp), nodes[i]["strategy"], nodes[i]["w_class"], r, (br, bp, cr, cp, t))/br
        fitness.append(f)
    return fitness


def proba_new_birth(selected, fitness, nodes, beta, h):
    """Compute the probability of imitation (with the fermi function)

    Args:
        beta (float): the intensity of selection
        fitness (list[float]): the estimated fitness of the 2 selected players (the resident and the invader)

    Returns:
        float: the probability that the resident imitates the invader
    """
    if nodes[selected[0]]["strategy"] == nodes[selected[1]]["strategy"]:
        return 0
    if nodes[selected[0]]["w_class"] != nodes[selected[1]]["w_class"]:
        if random.random() < h:
            return 0
    return fermi(beta, fitness[selected[0]], fitness[selected[1]])


def find_best(fitness):
    best_fit = 0
    id_best = []
    for i,f in enumerate(fitness):
        if f > best_fit:
            id_best.clear()
            id_best.append(i)
        elif f == best_fit:
            id_best.append(i)
    return random.choice(id_best)


def update(network, nodes:list, configuration, Z, beta, mu, h, r, payoffs_infos):
    """One step of the imitation process
    Update the nodes strategies

    Args:
        network (dict[int, list[int]]): the network
        nodes (list[int]): the list of nodes strategies
        configuration (dict[int, int]): the dictionary that links strategies to their distribution in the population
        Z (int): the population size
        A (list[list[float]]): the expected payoffs matrix
        beta (float): the intensity of selection

    Returns:
        list[int]: the updated list of nodes strategies
    """
    ir, ip = configuration
    fitness = estimate_fitness([i for i in range(Z)], network, nodes, r, payoffs_infos)
    selected = random.randint(0,Z-1)#find_best(fitness)
    selected_neighbour = random.choice(network[selected])
    if random.random() < proba_new_birth([selected, selected_neighbour], fitness, nodes, beta, h):
        nodes[selected_neighbour]["strategy"] = nodes[selected]["strategy"]
        if nodes[selected]["strategy"] == "Cooperate":
            if nodes[selected_neighbour]["w_class"] == "Rich":
                ir += 1
            else:
                ip += 1
        else:
            if nodes[selected_neighbour]["w_class"] == "Rich":
                ir -= 1
            else:
                ip -= 1
    elif random.random() < mu:
        s = random.choice(strategies)
        if s != nodes[selected_neighbour]["strategy"]:
            nodes[selected_neighbour]["strategy"] = s
            if s == "Cooperate":
                if nodes[selected_neighbour]["w_class"] == "Rich":
                    ir += 1
                else:
                    ip += 1
            else:
                if nodes[selected_neighbour]["w_class"] == "Rich":
                    ir -= 1
                else:
                    ip -= 1
    return ir, ip
            

# Parameters
nb_runs = 5                # number of runs: the number of independent realizations of the process
nb_generations = 100000      # number of generations to run the process
k_mean = N-1                # average connectivity

def compute_stationary_distribution(beta, mu, h, r, frac_ir, frac_ip):
    payoffs_infos = br, bp, cr, cp

    # Make the network
    graph = nx.barabasi_albert_graph(Z, k_mean // 2)
    network = transform_graph_to_dict_of_lists(graph)

    # Run the process
    all_stationary_distribution = []
    for i in range(nb_runs):
        print("\rrun number", i, end=" ", flush=True)
        #[{"strategy": random.choice(strategies), "w_class": "Rich"} for i in range(Zr)]
        ir = int(frac_ir*Zr)
        richs_C = [{"strategy": "Cooperate", "w_class": "Rich"} for i in range(ir)]
        richs_D = [{"strategy": "Defect", "w_class": "Rich"} for i in range(Zr-ir)]
        richs = richs_C
        richs.extend(richs_D)
        """
        ir = 0
        for ind in richs:
            if ind["strategy"] == "Cooperate":
                ir += 1
        poors = [{"strategy": random.choice(strategies), "w_class": "Poor"} for i in range(Zp)]
        ip = 0
        for ind in poors:
            if ind["strategy"] == "Cooperate":
                ip += 1
        """
        ip = int(frac_ip*Zp)
        poors_C = [{"strategy": "Cooperate", "w_class": "Poor"} for i in range(ip)]
        poors_D = [{"strategy": "Defect", "w_class": "Poor"} for i in range(Zp-ip)]
        poors = poors_C
        poors.extend(poors_D)
        nodes = richs
        nodes.extend(poors)
        random.shuffle(nodes)
        configuration = (ir, ip)
        
        stationary_distribution = np.zeros((Zp+1, Zr+1))

        for j in range(nb_generations):
            configuration = update(network, nodes, configuration, Z, beta, mu, h, r, payoffs_infos)
            ir, ip = configuration
            stationary_distribution[ip, ir] += 1
        stationary_distribution /= nb_generations
        all_stationary_distribution.append(stationary_distribution)
        print("final state", (ir, ip), Z-ir-ip)
    print("")
    all_stationary_distribution = np.array(all_stationary_distribution)
    mean_stationary_distribution = all_stationary_distribution.mean(0)
    return mean_stationary_distribution/mean_stationary_distribution.max()

p1 = compute_stationary_distribution(beta=3, mu=1/Z, h=0.0, r=0.2, frac_ir=0.1, frac_ip=0.1)
p2 = compute_stationary_distribution(beta=3, mu=1/Z, h=0.0, r=0.5, frac_ir=0.1, frac_ip=0.1)
p3 = compute_stationary_distribution(beta=3, mu=1/Z, h=1.0, r=0.2, frac_ir=0.1, frac_ip=0.1)


fig, axs = plt.subplots(1, 3)
axs[0].pcolormesh(p1, cmap=mpl.colormaps.get_cmap("binary"), shading="gouraud")
axs[0].axis("scaled")
axs[0].set_xlim(0,Zr)
axs[0].set_ylim(0,Zp)
axs[0].set_xlabel("ir")
axs[0].set_ylabel("ip")
axs[1].pcolormesh(p2, cmap=mpl.colormaps.get_cmap("binary"), shading="gouraud")
axs[1].axis("scaled")
axs[1].set_xlim(0,Zr)
axs[1].set_ylim(0,Zp)
axs[1].set_xlabel("ir")
axs[1].set_ylabel("ip")
m1 = axs[2].pcolormesh(p3, cmap=mpl.colormaps.get_cmap("binary"), shading="gouraud")
fig.colorbar(m1,ticks=[], label="stationary distribution (p)")
axs[2].axis("scaled")
axs[2].set_xlim(0,Zr)
axs[2].set_ylim(0,Zp)
axs[2].set_xlabel("ir")
axs[2].set_ylabel("ip")
plt.show()