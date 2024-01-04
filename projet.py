from math import (exp, comb)
import numpy as np
from matplotlib import pyplot as plt

Z = 200     # population size
Zr = 40     # number of richs in the population
Zp = 160    # number of poors in the population

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

r = 0.4     # risk perception


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
        double_sum = 0
        for jr in range(N-1):
            for jp in range(N-1-jr):
                double_sum += comb(ir, jr)*comb(ip, jp) * comb(Z-1-ir-ip, N-1-jr-jp) * payoffs((jr, jp), strategy, w_class, r)
        return double_sum/denominator
    else:
        if w_class == "Rich":
            double_sum = 0
            for jr in range(N-1):
                for jp in range(N-1-jr):
                    double_sum += comb(ir-1, jr)*comb(ip, jp) * comb(Z-ir-ip, N-1-jr-jp) * payoffs((jr+1, jp), strategy, w_class, r)
            return double_sum/denominator
        else:
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
    ir_C, ir_D, ip_C, ip_D = configuration # resp. number of rich cooperators, rich defectors, poors cooperators and poors defectors
    
    strat = selected_individual["strategy"]
    w_class = selected_individual["w_class"]
    
    fr_C = fitness([ir_C, ip_C], "Cooperate", "Rich", r)
    fr_D = fitness([ir_C, ip_C], "Defect", "Rich", r)
    fp_D = fitness([ir_C, ip_C], "Defect", "Poor", r)
    fp_C = fitness([ir_C, ip_C], "Cooperate", "Poor", r)
    
    if w_class == "Rich" and strat == "Cooperate":
        # fk_X = fr_C ; fk_Y = fr_D ; fl_Y = fp_D
        Tr_CD = (ir_C/Z) * ( (1-mu) * (ir_D/(Zr-1+(1-h)*Zp) * fermi(beta, fr_C, fr_D) +  ((1-h)*ip_D)/(Zr-1+(1-h)*Zp) * fermi(beta, fr_C, fp_D)) + mu )
        return Tr_CD
    elif w_class == "Poor" and strat == "Cooperate":
        # fk_X = fp_C ; fk_Y = fp_D ; fl_Y = fr_D
        Tp_CD = (ip_C/Z) * ( (1-mu) * (ip_D/(Zp-1+(1-h)*Zr) * fermi(beta, fp_C, fp_D) +  ((1-h)*ir_D)/(Zp-1+(1-h)*Zr) * fermi(beta, fp_C, fr_D)) + mu )
        return Tp_CD
    elif w_class == "Rich" and strat == "Defect":
        # fk_X = fr_D ; fk_Y = fr_C ; fl_Y = fp_C
        Tr_DC = (ir_D/Z) * ( (1-mu) * (ir_C/(Zr-1+(1-h)*Zp) * fermi(beta, fr_D, fr_C) +  ((1-h)*ip_C)/(Zr-1+(1-h)*Zp) * fermi(beta, fr_D, fp_C)) + mu )
        return Tr_DC
    elif w_class == "Poor" and strat == "Defect":
        # fk_X = fp_D ; fk_Y = fp_C ; fl_Y = fr_C
        Tp_DC = (ip_D/Z) * ( (1-mu) * (ip_C/(Zp-1+(1-h)*Zr) * fermi(beta, fp_D, fp_C) +  ((1-h)*ir_C)/(Zp-1+(1-h)*Zr) * fermi(beta, fp_D, fr_C)) + mu )
        return Tp_DC
    
    
def gradient_of_selection(configuration, mu:float, beta:float, h:float, r:float):
    Tir_pos = proba(configuration, {"strategy": "Defect", "w_class": "Rich"}, mu, beta, h, r)       # probability that a rich defector becomes a rich cooperator
    Tir_neg = proba(configuration, {"strategy": "Cooperate", "w_class": "Rich"}, mu, beta, h, r)    # probability that a rich cooperator becomes a rich defector
    
    Tip_pos = proba(configuration, {"strategy": "Defect", "w_class": "Poor"}, mu, beta, h, r)       # probability that a poor defector becomes a poor cooperator
    Tip_neg = proba(configuration, {"strategy": "Cooperate", "w_class": "Poor"}, mu, beta, h, r)    # probability that a poor cooperator becomes a poor defector
    
    nabla_i = [Tir_pos - Tir_neg, Tip_pos - Tip_neg]
    return nabla_i


def nabla_for_each_i(mu:float, beta:float, h:float, r:float):
    nabla_r = []
    nabla_p = []
    for ip in range(1, Zp):
        nr_ir = []
        np_ir = []
        for ir in range(1, Zr):
            configuration = (ir, Zr-ir, ip, Zp-ip) # ir_C, ir_D, ip_C, ip_D
            nabla_i = gradient_of_selection(configuration, mu, beta, h, r)
            nr_ir.append(nabla_i[0])
            np_ir.append(nabla_i[1])
        nabla_r.append(nr_ir)
        nabla_p.append(np_ir)
    return nabla_r, nabla_p


nabla_r, nabla_p = nabla_for_each_i(mu=1/Z, beta=3, h=0.0, r=0.2)

dx = np.array(nabla_r)
dy = np.array(nabla_p)

plt.figure("gradient selection")
plt.quiver(dx, dy)
plt.show()