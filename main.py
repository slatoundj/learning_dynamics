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


################
### Figure 1 ###
################
def compute_group_achivement_in_function_of_r(mu, beta, h, pop_infos, payoffs_infos, N):
    risk = [r/100 for r in range(101)]
    eta_G = []
    for r in risk:
        print("\r risk =", r, end=" ", flush=True)
        pi = compute_stationary_distribution(mu, beta, h, r, pop_infos, payoffs_infos, N)
        eta_G.append(eta_g(pi, r, pop_infos, grp_infos, N))
    print("")
    return risk, eta_G

rr = []
e = []
risk, etag = compute_group_achivement_in_function_of_r(mu=1/Z, beta=3, h=0.0, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
rr.append(risk)
e.append(etag)
risk, etag = compute_group_achivement_in_function_of_r(mu=1/Z, beta=3, h=1.0, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
rr.append(risk)
e.append(etag)
payoffs_infos = (1.0, 1.0, 0.1, 0.1, t)
risk, etag = compute_group_achivement_in_function_of_r(mu=1/Z, beta=3, h=0.0, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
rr.append(risk)
e.append(etag)
plot_grp_achievement_in_function_of_risk(e, rr)



##################
### Figure 2.A ###
##################
distrib = []
gradients = []
pi = compute_stationary_distribution(mu=1/Z, beta=3, h=0.0, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
grad = gradient_of_selection(mu=1/Z, beta=3, h=0.0, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
distrib.append(pi)
gradients.append(grad)
grp_achievement = eta_g(pi, r=0.2, pop_infos=pop_infos, grp_infos=grp_infos, N=N)
print("Group achievement =", grp_achievement)
#plot_gradient_with_distrib(gradient=grad, distribution=pi, p_max = 0.002, n_max=0.16, pop_infos=pop_infos)

##################
### Figure 2.B ###
##################
pi = compute_stationary_distribution(mu=1/Z, beta=3, h=0.7, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
grad = gradient_of_selection(mu=1/Z, beta=3, h=0.7, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
distrib.append(pi)
gradients.append(grad)
grp_achievement = eta_g(pi, r=0.2, pop_infos=pop_infos, grp_infos=grp_infos, N=N)
print("Group achievement =", grp_achievement)
#plot_gradient_with_distrib(gradient=grad, distribution=pi, p_max = 0.04, n_max=0.06, pop_infos=pop_infos)

##################
### Figure 2.C ###
##################
pi = compute_stationary_distribution(mu=1/Z, beta=3, h=1.0, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
grad = gradient_of_selection(mu=1/Z, beta=3, h=1.0, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
distrib.append(pi)
gradients.append(grad)
grp_achievement = eta_g(pi, r=0.2, pop_infos=pop_infos, grp_infos=grp_infos, N=N)
print("Group achievement =", grp_achievement)
#plot_gradient_with_distrib(gradient=grad, distribution=pi, p_max = 0.075, n_max=0.02, pop_infos=pop_infos)

plot_gradient_with_distrib3(gradients, distrib, [0.002, 0.04, 0.075], [0.16, 0.06, 0.02], pop_infos)


distrib = []
gradients = []
##################
### Figure 2.D ###
##################
pi = compute_stationary_distribution(mu=1/Z, beta=3, h=0.0, r=0.3, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
grad = gradient_of_selection(mu=1/Z, beta=3, h=0.0, r=0.3, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
grp_achievement = eta_g(pi, r=0.3, pop_infos=pop_infos, grp_infos=grp_infos, N=N)
distrib.append(pi)
gradients.append(grad)
print("Group achievement =", grp_achievement)
#plot_gradient_with_distrib(gradient=grad, distribution=pi, p_max = 0.003, n_max=0.16, pop_infos=pop_infos)

##################
### Figure 2.E ###
##################
pi = compute_stationary_distribution(mu=1/Z, beta=3, h=0.7, r=0.3, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
grad = gradient_of_selection(mu=1/Z, beta=3, h=0.7, r=0.3, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
distrib.append(pi)
gradients.append(grad)
grp_achievement = eta_g(pi, r=0.3, pop_infos=pop_infos, grp_infos=grp_infos, N=N)
print("Group achievement =", grp_achievement)
#plot_gradient_with_distrib(gradient=grad, distribution=pi, p_max = 0.002, n_max=0.06, pop_infos=pop_infos)

##################
### Figure 2.F ###
##################
pi = compute_stationary_distribution(mu=1/Z, beta=3, h=1.0, r=0.3, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
grad = gradient_of_selection(mu=1/Z, beta=3, h=1.0, r=0.3, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
distrib.append(pi)
gradients.append(grad)
grp_achievement = eta_g(pi, r=0.3, pop_infos=pop_infos, grp_infos=grp_infos, N=N)
print("Group achievement =", grp_achievement)
#plot_gradient_with_distrib(gradient=grad, distribution=pi, p_max = 0.020, n_max=0.03, pop_infos=pop_infos)

plot_gradient_with_distrib3(gradients, distrib, [0.003, 0.02, 0.020], [0.16, 0.06, 0.03], pop_infos)


distrib = []
gradients = []
##################
### Figure 3.A ###
##################
pi = compute_stationary_distribution(mu=1/Z, beta=5, h=1.0, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
grad = gradient_of_selection(mu=1/Z, beta=5, h=1.0, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N)
distrib.append(pi)
gradients.append(grad)
grp_achievement = eta_g(pi, r=0.2, pop_infos=pop_infos, grp_infos=grp_infos, N=N)
print("Group achievement =", grp_achievement)
#plot_gradient_with_distrib(gradient=grad, distribution=pi, p_max = 0.076, n_max=0.03, pop_infos=pop_infos)

##################
### Figure 3.B ###
##################
obstination_info = (1/10, "Rich")
pi = compute_stationary_distribution(mu=1/Z, beta=5, h=1.0, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N, obstination=True, obstination_info=obstination_info)
grad = gradient_of_selection(mu=1/Z, beta=5, h=1.0, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N, obstination=True, obstination_info=obstination_info)
distrib.append(pi)
gradients.append(grad)
grp_achievement = eta_g(pi, r=0.2, pop_infos=pop_infos, grp_infos=grp_infos, N=N, obstination=True, obstination_info=obstination_info)
print("Group achievement =", grp_achievement)
#plot_gradient_with_distrib(gradient=grad, distribution=pi, p_max = 0.004, n_max=0.03, pop_infos=pop_infos, obstination=True, obstination_info=obstination_info)

##################
### Figure 3.B ###
##################
obstination_info = (1/10, "Poor")
pi = compute_stationary_distribution(mu=1/Z, beta=5, h=1.0, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N, obstination=True, obstination_info=obstination_info, rtol=1e-3)
grad = gradient_of_selection(mu=1/Z, beta=5, h=1.0, r=0.2, pop_infos=pop_infos, payoffs_infos=payoffs_infos, N=N, obstination=True, obstination_info=obstination_info)
distrib.append(pi)
gradients.append(grad)
grp_achievement = eta_g(pi, r=0.2, pop_infos=pop_infos, grp_infos=grp_infos, N=N, obstination=True, obstination_info=obstination_info)
print("Group achievement =", grp_achievement)
#plot_gradient_with_distrib(gradient=grad, distribution=pi, p_max = 0.002, n_max=0.04, pop_infos=pop_infos, obstination=True, obstination_info=obstination_info)

plot_gradient_with_distrib3(gradients, distrib, [0.076, 0.004, 0.002], [0.03, 0.03, 0.04], pop_infos, obstination=True, obstination_info=[None, (1/10, "Rich"), (1/10, "Poor")])
