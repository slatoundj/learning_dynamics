from math import comb

from utils import (heaviside, V)


def gr_achievement(j_config, grp_infos):
    (jr, jp) = j_config
    cr, cp, t = grp_infos
    delta = cr*jr + cp*jp - t
    return heaviside(delta)


def gr_achievement_over_pop(i_config, strategy, w_class, pop_infos, grp_infos, N):
    (ir, ip) = i_config
    Zr, Zp, Z = pop_infos
    
    if ir == Zr and ip == Zp:
        return 1
    
    n = []
    ns = 0
    if strategy == "Defect":
        # Defectors
        for jr in range(N-1):
            for jp in range(N-1-jr):
                    val = comb(ir, jr)*comb(ip, jp) * comb(Z-1-ir-ip, N-1-jr-jp)
                    n.append(val * gr_achievement((jr, jp), grp_infos))
                    ns += val
        if ns == 0:
            return 0
        return sum(n)/ns
    else:
        if w_class == "Rich":
            # Rich cooperator
            if ir == 0:
                return 0    # If there is no rich cooperator in the population, their fitness is 0 (unknown)
            for jr in range(N-1):
                for jp in range(N-1-jr):
                    val = comb(ir-1, jr)*comb(ip, jp) * comb(Z-ir-ip, N-1-jr-jp)
                    n.append(val*gr_achievement((jr+1, jp), grp_infos))
                    ns += val
            if ns == 0:
                return 0
            return sum(n)/ns
        else:
            # Poor cooperator
            if ip == 0:
                return 0    # If there is no poor cooperator in the population, their fitness is 0 (unknown)
            for jr in range(N-1):
                for jp in range(N-1-jr):
                    val = comb(ir, jr)*comb(ip-1, jp) * comb(Z-ir-ip, N-1-jr-jp)
                    n.append(val*gr_achievement((jr, jp+1), grp_infos))
                    ns += val
            if ns == 0:
                return 0
            return sum(n)/ns
        
        
def aG(i, pop_infos, grp_infos, N):
    Zr, Zp, Z = pop_infos
    (ir, ip) = i
    ag_i = ir * gr_achievement_over_pop((ir, ip), "Cooperate", "Rich", pop_infos, grp_infos, N)
    ag_i += (Zr - ir) * gr_achievement_over_pop((ir, ip), "Defect", "Rich", pop_infos, grp_infos, N)
    ag_i += ip * gr_achievement_over_pop((ir, ip), "Cooperate", "Poor", pop_infos, grp_infos, N)
    ag_i += (Zp - ip) * gr_achievement_over_pop((ir, ip), "Defect", "Poor", pop_infos, grp_infos, N)
    return ag_i/Z


def eta_g(stationary_distribution, r, pop_infos, grp_infos, N, obstination=False, obstination_info=None):
    Zr, Zp, Z = pop_infos
    ir_min = 0
    ip_min = 0
    if obstination and obstination_info != None:
        frac, w_class = obstination_info
        if w_class == "Rich":
            ir_min = int(frac*Zr)
        elif w_class == "Poor":
            ip_min = int(frac*Zp)
    eta_g_i = 0
    for ir in range(ir_min, Zr+1):
        for ip in range(ip_min, Zp+1):
            i = V((ir-ir_min, ip-ip_min), Zr-ir_min)
            eta_g_i += stationary_distribution[i] * aG((ir, ip), pop_infos, grp_infos, N)
    return eta_g_i


#print(aG((40,160)))
#print(gr_achievement_over_pop((1, 19), "Cooperate", "Rich"))