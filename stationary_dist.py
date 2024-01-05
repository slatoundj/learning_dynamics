from math import (floor, exp, comb)
import numpy as np
from matplotlib import pyplot as plt
import time
import scipy as sc

start = time.time()

Zr = 40
Zp = 160
Z = Zr + Zp

h = 0       # homophily

# initial endowment
br = 2.5    
bp = 0.625

# average endowment
avg_b = 0.2*br + 0.8*0.625

# cooperation cost
c = 0.1
cr = c * br
cp = c * bp

N = 6   # sub group size
M = 3   # 
threshold = M*c*avg_b

strategies = ["Defect", "Cooperate"]
w_classes = ["Rich", "Poor"]

mu = 1/Z    # mutation probability
beta = 3    # intensity of selection

r = 0.2     # risk perception

numStates = 6601


def heaviside(x):
    if x >= 0:
        return 1
    else:
        return 0


def payoffs(j_config, strategy, w_class, r):
    (jr, jp) = j_config
    
    delta = cr*jr + cp*jp - threshold
    
    pi_D_r = br * heaviside(delta) + (1-r) * (1 - heaviside(delta))
    pi_D_p = bp * heaviside(delta) + (1-r) * (1 - heaviside(delta))
    
    pi_C_r = pi_D_r - cr
    pi_C_p = pi_D_p - cp
    
    if strategy == "Defect":
        if w_class == "Rich":
            return pi_D_r
        else:
            return pi_D_p
    else:
        if w_class == "Rich":
            return pi_C_r
        else:
            return pi_C_p


def fitness(i_config, strategy, w_class, r):
    (ir, ip) = i_config
    
    denominator = comb(Z-1, N-1)
    
    if strategy == "Defect":
        # Defectors
        if ir == Zr and ip == Zp:
            return 0    # If there is no defector in the population, their fitness is 0 (unknown)
        double_sum = 0
        for jr in range(N-1):
            for jp in range(N-1-jr):
                double_sum += comb(ir, jr)*comb(ip, jp) * comb(Z-1-ir-ip, N-1-jr-jp) * payoffs((jr, jp), strategy, w_class, r)
        return double_sum/denominator
    else:
        if w_class == "Rich":
            # Rich cooperator
            if ir == 0:
                return 0    # If there is no rich cooperator in the population, their fitness is 0 (unknown)
            double_sum = 0
            for jr in range(N-1):
                for jp in range(N-1-jr):
                    double_sum += comb(ir-1, jr)*comb(ip, jp) * comb(Z-ir-ip, N-1-jr-jp) * payoffs((jr+1, jp), strategy, w_class, r)
            return double_sum/denominator
        else:
            # Poor cooperator
            if ip == 0:
                return 0    # If there is no poor cooperator in the population, their fitness is 0 (unknown)
            double_sum = 0
            for jr in range(N-1):
                for jp in range(N-1-jr):
                    double_sum += comb(ir, jr)*comb(ip-1, jp) * comb(Z-ir-ip, N-1-jr-jp) * payoffs((jr, jp+1), strategy, w_class, r)
            return double_sum/denominator


def fermi(beta, fx, fy):
    return 1 / (1 + exp(beta * (fx - fy)))


def proba(configuration, selected_individual, mu:float, beta:float, h:float, r:float):
    """Compute the probability that the invader imitates the strategy of the resident (with invader and resident that have different strategy)

    Args:
        configuration (_type_): the number of rich cooperators, rich defectors, poors cooperators and poors defectors in the population
        individuals (_type_): _description_
        mu (float): _description_
        beta (float): _description_
        h (float): _description_

    Returns:
        _type_: _description_
    """
    ir_C, ip_C = configuration # resp. number of rich cooperators, rich defectors, poors cooperators and poors defectors
    ir_D = Zr - ir_C
    ip_D = Zp - ip_C
    
    strat = selected_individual["strategy"]
    w_class = selected_individual["w_class"]
    
    if w_class == "Rich" and strat == "Cooperate":
        # fk_X = fr_C ; fk_Y = fr_D ; fl_Y = fp_D
        if ir_C == 0:
            return 0
        else:
            fr_C = fitness([ir_C, ip_C], "Cooperate", "Rich", r)
            fr_D = fitness([ir_C, ip_C], "Defect", "Rich", r)
            fp_D = fitness([ir_C, ip_C], "Defect", "Poor", r)
            Tr_CD = (ir_C/Z) * ( (1-mu) * (ir_D/(Zr-1+(1-h)*Zp) * fermi(beta, fr_C, fr_D) +  ((1-h)*ip_D)/(Zr-1+(1-h)*Zp) * fermi(beta, fr_C, fp_D)) + mu )
            return Tr_CD
    elif w_class == "Poor" and strat == "Cooperate":
        # fk_X = fp_C ; fk_Y = fp_D ; fl_Y = fr_D
        if ip_C == 0:
            return 0
        else:
            fp_C = fitness([ir_C, ip_C], "Cooperate", "Poor", r)
            fp_D = fitness([ir_C, ip_C], "Defect", "Poor", r)
            fr_D = fitness([ir_C, ip_C], "Defect", "Rich", r)
            Tp_CD = (ip_C/Z) * ( (1-mu) * (ip_D/(Zp-1+(1-h)*Zr) * fermi(beta, fp_C, fp_D) +  ((1-h)*ir_D)/(Zp-1+(1-h)*Zr) * fermi(beta, fp_C, fr_D)) + mu )
            return Tp_CD
    elif w_class == "Rich" and strat == "Defect":
        # fk_X = fr_D ; fk_Y = fr_C ; fl_Y = fp_C
        if ir_D == 0:
            return 0
        else:
            fr_D = fitness([ir_C, ip_C], "Defect", "Rich", r)
            fr_C = fitness([ir_C, ip_C], "Cooperate", "Rich", r)
            fp_C = fitness([ir_C, ip_C], "Cooperate", "Poor", r)
            Tr_DC = (ir_D/Z) * ( (1-mu) * (ir_C/(Zr-1+(1-h)*Zp) * fermi(beta, fr_D, fr_C) +  ((1-h)*ip_C)/(Zr-1+(1-h)*Zp) * fermi(beta, fr_D, fp_C)) + mu )
            return Tr_DC
    elif w_class == "Poor" and strat == "Defect":
        # fk_X = fp_D ; fk_Y = fp_C ; fl_Y = fr_C
        if ip_D == 0:
            return 0
        else:
            fp_D = fitness([ir_C, ip_C], "Defect", "Poor", r)
            fp_C = fitness([ir_C, ip_C], "Cooperate", "Poor", r)
            fr_C = fitness([ir_C, ip_C], "Cooperate", "Rich", r)
            Tp_DC = (ip_D/Z) * ( (1-mu) * (ip_C/(Zp-1+(1-h)*Zr) * fermi(beta, fp_D, fp_C) +  ((1-h)*ir_C)/(Zp-1+(1-h)*Zr) * fermi(beta, fp_D, fr_C)) + mu )
            return Tp_DC
    

"""
0 à 40 coopérateurs riches
0 à 160 coopérateurs pauvres
"""


def V(i_config):
    ir, ip = i_config
    return ir + ip + ip * Zr


def rec_V(x):
    ip = floor(x/Zr)
    ir = x - ip * Zr - ip
    while ir < 0:
        ip -= 1
        ir = x - ip * Zr - ip
    return (ir, ip)   

"""
W p - lambda I p = 0
(W - lambda I) p = 0
"""

"""
P t - lambda I t = 0
(P - lambda I) t = 0
"""

def null(a, rtol=1e-8):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()


def transition_matrix(mu, beta, h, r):
    w_matrix = np.zeros((numStates, numStates))
    # q:row ; p:column
    for q in range(numStates):
        q_config = rec_V(q)
        q_ir, q_ip = q_config
        for p in range(numStates):
            if q == 0 and p == 0:
                pr1 = proba(q_config, {"strategy": "Defect", "w_class": "Rich"}, mu, beta, h, r)        # Probability that state (ir, ip) switch to state (ir+1, ip)
                pr2 = proba(q_config, {"strategy": "Defect", "w_class": "Poor"}, mu, beta, h, r)        # Probability that state (ir, ip) switch to state (ir, ip+1)
                w_matrix[q, p] = 1 - pr1 - pr2    
            elif p == q + 1:
                w_matrix[q, p] = proba(q_config, {"strategy": "Defect", "w_class": "Rich"}, mu, beta, h, r)     # Probability that state (ir, ip) switch to state (ir+1, ip)
            elif p == q + Zr + 1:
                w_matrix[q, p] = proba(q_config, {"strategy": "Defect", "w_class": "Poor"}, mu, beta, h, r)     # Probability that state (ir, ip) switch to state (ir, ip+1)
            elif p == q - 1:
                w_matrix[q, p] = proba(q_config, {"strategy": "Cooperate", "w_class": "Rich"}, mu, beta, h, r)  # Probability that state (ir, ip) switch to state (ir-1, ip)
            elif p == q - Zr - 1:
                w_matrix[q, p] = proba(q_config, {"strategy": "Cooperate", "w_class": "Poor"}, mu, beta, h, r)  # Probability that state (ir, ip) switch to state (ir, ip-1)
                
            elif p == q:
                pr1 = proba(q_config, {"strategy": "Defect", "w_class": "Rich"}, mu, beta, h, r)        # Probability that state (ir, ip) switch to state (ir+1, ip)
                pr2 = proba(q_config, {"strategy": "Defect", "w_class": "Poor"}, mu, beta, h, r)        # Probability that state (ir, ip) switch to state (ir, ip+1)
                pr3 = proba(q_config, {"strategy": "Cooperate", "w_class": "Rich"}, mu, beta, h, r)     # Probability that state (ir, ip) switch to state (ir-1, ip)
                pr4 = proba(q_config, {"strategy": "Cooperate", "w_class": "Poor"}, mu, beta, h, r)     # Probability that state (ir, ip) switch to state (ir, ip-1)
                w_matrix[q, p] = 1 - pr1 - pr2 - pr3 - pr4
        #print("\r", q, end=" ", flush=True)
    #print("")
    return w_matrix
        
"""
P = transition_matrix(1/Z, 3, h=0.7, r=0.3)

Q = (np.eye(P.shape[0]) - P).transpose()
rank, null_space = null(Q)
print(null_space.shape)
pi = np.abs(null_space[:,0])
pi = pi/pi.sum()
print(pi.sum())

plt.figure("stationary distrib")
x = np.arange(pi.shape[0])
plt.plot(x, pi)
plt.show()

matrix = np.zeros((Zp+1, Zr+1))
for i in range(numStates):
    ir, ip = rec_V(i)
    matrix[ip, ir] = pi[i]
    
plt.matshow(matrix)

m = matrix

#Define a colormap for the plot
cmap = plt.cm.get_cmap('gray')

#Normalize the values in the matrix to map to colors
norm = plt.Normalize(m.min(), m.max())

#Create a scatter plot
fig, ax = plt.subplots()
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        color = (m[i][j] - m.min())/(m.max()-m.min())
        ax.scatter(j, i, c=[[1-color,1-color,1-color]], alpha=1, s=2)  # j and i are reversed to match matrix indexing

#Set the aspect ratio to make sure dots are square
ax.set_aspect('equal')

#Show the plot
plt.show()
"""

def compute_stationary_distribution(mu, beta, h, r):
    P = transition_matrix(mu, beta, h, r)
    Q = (np.eye(P.shape[0]) - P).transpose()
    rank, null_space = null(Q)
    pi = np.abs(null_space[:,0])
    pi = pi/pi.sum()
    return pi


def aG(i, r):
    (ir, ip) = i
    ag_i = ir * fitness((ir, ip), "Cooperate", "Rich", r)
    ag_i += (Zr-ir) * fitness((ir, ip), "Defect", "Rich", r)
    ag_i += ip * fitness((ir, ip), "Cooperate", "Poor", r)
    ag_i += (Zp - ip) * fitness((ir, ip), "Defect", "Poor", r)
    return ag_i/Z


def eta_g(stationary_distribution, r):
    eta_g_i = 0
    for ir in range(Zr+1):
        for ip in range(Zp+1):
            i = V((ir, ip))
            eta_g_i += stationary_distribution[i] * aG((ir, ip), r)
    return eta_g_i/numStates


def compute_group_achivement_in_function_of_r(mu, beta, h):
    risk = [r/100 for r in range(101)]
    eta_G = []
    for r in risk:
        pi = compute_stationary_distribution(mu, beta, h, r)
        eta_G.append(eta_g(pi, r))
    return risk, eta_G


risk, eta_G = compute_group_achivement_in_function_of_r(mu=1/Z, beta=3, h=0.0)

plt.figure("Group achievement in function of risk")
plt.plot(risk, eta_G)


end = time.time()
print("elapsed time =", end-start, "seconds")