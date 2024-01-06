import numpy as np

from probabilities import proba
from utils import (V, rec_V, null, null_cpu)


def transition_matrix(mu, beta, h, r, pop_infos, payoffs_infos, N, obstination=False, obstination_info=None):
    Zr, Zp, Z = pop_infos
    ir_min = 0
    ip_min = 0
    if obstination and obstination_info != None:
        frac, w_class = obstination_info
        if w_class == "Rich":
            ir_min = int(frac*Zr)
        elif w_class == "Poor":
            ip_min = int(frac*Zp)
    Zr = Zr - ir_min    # pseudo Zr
    Zp = Zp - ip_min    # pseudo Zp
    numStates = int((Zr+1) * (Zp+1))
    w_matrix = np.zeros((numStates, numStates))
    # q:row ; p:column
    for q in range(numStates):
        q_ir, q_ip = rec_V(q, Zr)
        q_ir += ir_min
        q_ip += ip_min
        q_config = (q_ir, q_ip)
        for p in range(numStates):
            if q == 0 and p == 0:
                pr1 = proba(q_config, {"strategy": "Defect", "w_class": "Rich"}, mu, beta, h, r, pop_infos, payoffs_infos, N)        # Probability that state (ir, ip) switch to state (ir+1, ip)
                pr2 = proba(q_config, {"strategy": "Defect", "w_class": "Poor"}, mu, beta, h, r, pop_infos, payoffs_infos, N)        # Probability that state (ir, ip) switch to state (ir, ip+1)
                w_matrix[q, p] = 1 - pr1 - pr2    
            elif p == q + 1:
                w_matrix[q, p] = proba(q_config, {"strategy": "Defect", "w_class": "Rich"}, mu, beta, h, r, pop_infos, payoffs_infos, N)     # Probability that state (ir, ip) switch to state (ir+1, ip)
            elif p == q + Zr + 1:
                w_matrix[q, p] = proba(q_config, {"strategy": "Defect", "w_class": "Poor"}, mu, beta, h, r, pop_infos, payoffs_infos, N)     # Probability that state (ir, ip) switch to state (ir, ip+1)
            elif p == q - 1:
                w_matrix[q, p] = proba(q_config, {"strategy": "Cooperate", "w_class": "Rich"}, mu, beta, h, r, pop_infos, payoffs_infos, N)  # Probability that state (ir, ip) switch to state (ir-1, ip)
            elif p == q - Zr - 1:
                w_matrix[q, p] = proba(q_config, {"strategy": "Cooperate", "w_class": "Poor"}, mu, beta, h, r, pop_infos, payoffs_infos, N)  # Probability that state (ir, ip) switch to state (ir, ip-1)
                
            elif p == q:
                pr1 = proba(q_config, {"strategy": "Defect", "w_class": "Rich"}, mu, beta, h, r, pop_infos, payoffs_infos, N)        # Probability that state (ir, ip) switch to state (ir+1, ip)
                pr2 = proba(q_config, {"strategy": "Defect", "w_class": "Poor"}, mu, beta, h, r, pop_infos, payoffs_infos, N)        # Probability that state (ir, ip) switch to state (ir, ip+1)
                pr3 = proba(q_config, {"strategy": "Cooperate", "w_class": "Rich"}, mu, beta, h, r, pop_infos, payoffs_infos, N)     # Probability that state (ir, ip) switch to state (ir-1, ip)
                pr4 = proba(q_config, {"strategy": "Cooperate", "w_class": "Poor"}, mu, beta, h, r, pop_infos, payoffs_infos, N)     # Probability that state (ir, ip) switch to state (ir, ip-1)
                w_matrix[q, p] = 1 - pr1 - pr2 - pr3 - pr4
    return w_matrix


def compute_stationary_distribution(mu, beta, h, r, pop_infos, payoffs_infos, N, obstination=False, obstination_info=None, rtol=1e-6, on_cpu=False):
    P = transition_matrix(mu, beta, h, r, pop_infos, payoffs_infos, N, obstination, obstination_info)
    Q = (np.eye(P.shape[0]) - P).transpose()
    if on_cpu:
        rank, null_space = null_cpu(Q, rtol)
    else:
        rank, null_space = null(Q, rtol)
    pi = np.abs(null_space[:,0])
    pi = pi/pi.sum()
    return pi