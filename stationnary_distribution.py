import numpy as np

from probabilities import proba
from utils import (rec_V, null)


def transition_matrix(mu, beta, h, r, pop_infos, payoffs_infos, N):
    Zr, Zp, Z = pop_infos
    numStates = (Zr+1) * (Zp+1)
    w_matrix = np.zeros((numStates, numStates))
    # q:row ; p:column
    for q in range(numStates):
        q_config = rec_V(q, Zr)
        q_ir, q_ip = q_config
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


def compute_stationary_distribution(mu, beta, h, r, pop_infos, payoffs_infos, N):
    P = transition_matrix(mu, beta, h, r, pop_infos, payoffs_infos, N)
    Q = (np.eye(P.shape[0]) - P).transpose()
    rank, null_space = null(Q)
    pi = np.abs(null_space[:,0])
    pi = pi/pi.sum()
    return pi