from math import (comb, exp)

from utils import (fermi, heaviside)


def payoffs(j_config: tuple, strategy: str, w_class: str, r: float, payoffs_infos: tuple):
    (jr, jp) = j_config
    br, bp, cr, cp, t = payoffs_infos
    
    delta = cr*jr + cp*jp - t
    
    pi_D_r = br * (heaviside(delta) + (1-r) * (1 - heaviside(delta)))
    pi_D_p = bp * (heaviside(delta) + (1-r) * (1 - heaviside(delta)))
    
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


def fitness(i_config: tuple, strategy: str, w_class: str, r: float, pop_infos: tuple, payoffs_infos: tuple, N: float):
    (ir, ip) = i_config
    Zr, Zp, Z = pop_infos
    
    denominator = comb(Z-1, N-1)
    
    if strategy == "Defect":
        # Defectors
        if ir == Zr and ip == Zp:
            return 0    # If there is no defector in the population, their fitness is 0 (unknown)
        double_sum = 0
        for jr in range(N-1):
            for jp in range(N-1-jr):
                double_sum += comb(ir, jr)*comb(ip, jp) * comb(Z-1-ir-ip, N-1-jr-jp) * payoffs((jr, jp), strategy, w_class, r, payoffs_infos)
        return double_sum/denominator
    else:
        if w_class == "Rich":
            # Rich cooperator
            if ir == 0:
                return 0    # If there is no rich cooperator in the population, their fitness is 0 (unknown)
            double_sum = 0
            for jr in range(N-1):
                for jp in range(N-1-jr):
                    double_sum += comb(ir-1, jr)*comb(ip, jp) * comb(Z-ir-ip, N-1-jr-jp) * payoffs((jr+1, jp), strategy, w_class, r, payoffs_infos)
            return double_sum/denominator
        else:
            # Poor cooperator
            if ip == 0:
                return 0    # If there is no poor cooperator in the population, their fitness is 0 (unknown)
            double_sum = 0
            for jr in range(N-1):
                for jp in range(N-1-jr):
                    double_sum += comb(ir, jr)*comb(ip-1, jp) * comb(Z-ir-ip, N-1-jr-jp) * payoffs((jr, jp+1), strategy, w_class, r, payoffs_infos)
            return double_sum/denominator


def proba(configuration: tuple, selected: dict, mu:float, beta:float, h:float, r:float, pop_infos: tuple, payoffs_infos: tuple, N: int):
    Zr, Zp, Z = pop_infos
    ir_C, ip_C = configuration # resp. number of rich cooperators, rich defectors, poors cooperators and poors defectors
    ir_D = Zr - ir_C
    ip_D = Zp - ip_C
    
    strat = selected["strategy"]
    w_class = selected["w_class"]
    
    if w_class == "Rich" and strat == "Cooperate":
        # fk_X = fr_C ; fk_Y = fr_D ; fl_Y = fp_D
        if ir_C == 0:
            return 0
        else:
            fr_C = fitness([ir_C, ip_C], "Cooperate", "Rich", r, pop_infos, payoffs_infos, N)
            fr_D = fitness([ir_C, ip_C], "Defect", "Rich", r, pop_infos, payoffs_infos, N)
            fp_D = fitness([ir_C, ip_C], "Defect", "Poor", r, pop_infos, payoffs_infos, N)
            Tr_CD = (ir_C/Z) * ( (1-mu) * (ir_D/(Zr-1+(1-h)*Zp) * fermi(beta, fr_C, fr_D) +  ((1-h)*ip_D)/(Zr-1+(1-h)*Zp) * fermi(beta, fr_C, fp_D)) + mu )
            return Tr_CD
    elif w_class == "Poor" and strat == "Cooperate":
        # fk_X = fp_C ; fk_Y = fp_D ; fl_Y = fr_D
        if ip_C == 0:
            return 0
        else:
            fp_C = fitness([ir_C, ip_C], "Cooperate", "Poor", r, pop_infos, payoffs_infos, N)
            fp_D = fitness([ir_C, ip_C], "Defect", "Poor", r, pop_infos, payoffs_infos, N)
            fr_D = fitness([ir_C, ip_C], "Defect", "Rich", r, pop_infos, payoffs_infos, N)
            Tp_CD = (ip_C/Z) * ( (1-mu) * (ip_D/(Zp-1+(1-h)*Zr) * fermi(beta, fp_C, fp_D) +  ((1-h)*ir_D)/(Zp-1+(1-h)*Zr) * fermi(beta, fp_C, fr_D)) + mu )
            return Tp_CD
    elif w_class == "Rich" and strat == "Defect":
        # fk_X = fr_D ; fk_Y = fr_C ; fl_Y = fp_C
        if ir_D == 0:
            return 0
        else:
            fr_D = fitness([ir_C, ip_C], "Defect", "Rich", r, pop_infos, payoffs_infos, N)
            fr_C = fitness([ir_C, ip_C], "Cooperate", "Rich", r, pop_infos, payoffs_infos, N)
            fp_C = fitness([ir_C, ip_C], "Cooperate", "Poor", r, pop_infos, payoffs_infos, N)
            Tr_DC = (ir_D/Z) * ( (1-mu) * (ir_C/(Zr-1+(1-h)*Zp) * fermi(beta, fr_D, fr_C) +  ((1-h)*ip_C)/(Zr-1+(1-h)*Zp) * fermi(beta, fr_D, fp_C)) + mu )
            return Tr_DC
    elif w_class == "Poor" and strat == "Defect":
        # fk_X = fp_D ; fk_Y = fp_C ; fl_Y = fr_C
        if ip_D == 0:
            return 0
        else:
            fp_D = fitness([ir_C, ip_C], "Defect", "Poor", r, pop_infos, payoffs_infos, N)
            fp_C = fitness([ir_C, ip_C], "Cooperate", "Poor", r, pop_infos, payoffs_infos, N)
            fr_C = fitness([ir_C, ip_C], "Cooperate", "Rich", r, pop_infos, payoffs_infos, N)
            Tp_DC = (ip_D/Z) * ( (1-mu) * (ip_C/(Zp-1+(1-h)*Zr) * fermi(beta, fp_D, fp_C) +  ((1-h)*ir_C)/(Zp-1+(1-h)*Zr) * fermi(beta, fp_D, fr_C)) + mu )
            return Tp_DC