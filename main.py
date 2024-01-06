from stationnary_distribution import compute_stationary_distribution
from gradient_selection import gradient_of_selection
from group_achievement import eta_g
from plots import (plot_distribution, plot_distribution_grid)


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
pi = compute_stationary_distribution(mu=1/Z, beta=3, h=1.0, r=0.39, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
print(pi.sum())
plot_distribution_grid(pi, pop_infos)
