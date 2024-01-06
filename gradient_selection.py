from probabilities import proba


def gradient_i(configuration, mu:float, beta:float, h:float, r:float, pop_infos, payoffs_infos, N):
    Tir_pos = proba(configuration, {"strategy": "Defect", "w_class": "Rich"}, mu, beta, h, r, pop_infos, payoffs_infos, N)       # probability that a rich defector becomes a rich cooperator
    Tir_neg = proba(configuration, {"strategy": "Cooperate", "w_class": "Rich"}, mu, beta, h, r, pop_infos, payoffs_infos, N)    # probability that a rich cooperator becomes a rich defector
    
    Tip_pos = proba(configuration, {"strategy": "Defect", "w_class": "Poor"}, mu, beta, h, r, pop_infos, payoffs_infos, N)       # probability that a poor defector becomes a poor cooperator
    Tip_neg = proba(configuration, {"strategy": "Cooperate", "w_class": "Poor"}, mu, beta, h, r, pop_infos, payoffs_infos, N)    # probability that a poor cooperator becomes a poor defector
    
    nabla_i = [Tir_pos - Tir_neg, Tip_pos - Tip_neg]
    return nabla_i


def gradient_of_selection(mu:float, beta:float, h:float, r:float, pop_infos, payoffs_infos, N):
    Zr, Zp, Z = pop_infos
    nabla_r = []
    nabla_p = []
    for ip in range(0, Zp+1):
        nr_ir = []
        np_ir = []
        for ir in range(0, Zr+1):
            nabla_i = gradient_i((ir, ip), mu, beta, h, r, pop_infos, payoffs_infos, N)
            nr_ir.append(nabla_i[0])
            np_ir.append(nabla_i[1])
        nabla_r.append(nr_ir)
        nabla_p.append(np_ir)
    return nabla_r, nabla_p