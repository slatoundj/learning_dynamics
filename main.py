from stationnary_distribution import compute_stationary_distribution
from gradient_selection import gradient_of_selection
from group_achievement import eta_g
from plots import *


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


pop_infos = (Zr, Zp, Z)
payoffs_infos = (br, bp, cr, cp, t)
grp_infos = (cr, cp, t)


def compute_group_achivement_in_function_of_r(mu, beta, h):
    risk = [0.0, 0.5, 1.0] #[r/100 for r in range(101)]
    eta_G = []
    for r in risk:
        print("\r risk =", r, end=" ", flush=True)
        pi = compute_stationary_distribution(mu, beta, h, r, pop_infos, payoffs_infos, N)
        print(pi.sum())
        eta_G.append(eta_g(pi, r, pop_infos, grp_infos, N))
    print("")
    return risk, eta_G


#pi = compute_stationary_distribution(mu=1/Z, beta=3, h=1.0, r=0.39, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
#print(pi.sum())
#plot_distribution_grid(pi, pop_infos)


#risk, etag = compute_group_achivement_in_function_of_r(mu=1/Z, beta=3, h=0.5)
#plot_grp_achievement_in_function_of_risk([etag], [risk])

##################
### Figure 2.A ###
##################
"""
pi = compute_stationary_distribution(mu=1/Z, beta=3, h=0.0, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
grad = gradient_of_selection(mu=1/Z, beta=3, h=0.0, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
grp_achievement = eta_g(pi, r=0.2, pop_infos=pop_infos, grp_infos=grp_infos, N=N)
print("Group achievement =", grp_achievement)
plot_gradient_with_distrib(gradient=grad, distribution=pi, p_max = 0.002, pop_infos=pop_infos)
"""

##################
### Figure 2.B ###
##################
"""
pi = compute_stationary_distribution(mu=1/Z, beta=3, h=0.7, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
grad = gradient_of_selection(mu=1/Z, beta=3, h=0.7, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
grp_achievement = eta_g(pi, r=0.2, pop_infos=pop_infos, grp_infos=grp_infos, N=N)
print("Group achievement =", grp_achievement)
plot_gradient_with_distrib(gradient=grad, distribution=pi, p_max = 0.04, pop_infos=pop_infos)
"""


##################
### Figure 2.C ###
##################
pi = compute_stationary_distribution(mu=1/Z, beta=3, h=1.0, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
grad = gradient_of_selection(mu=1/Z, beta=3, h=1.0, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
grp_achievement = eta_g(pi, r=0.2, pop_infos=pop_infos, grp_infos=grp_infos, N=N)
print("Group achievement =", grp_achievement)
plot_gradient_with_distrib(gradient=grad, distribution=pi, p_max = 0.075, pop_infos=pop_infos)
