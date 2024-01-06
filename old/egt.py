import egttools as egt
import numpy as np

Z = 200
cost = 0.1
M = 3

DEFECT = 0
COOPERATE = 1

strategies = [egt.behaviors.pgg_behaviors.PGGOneShotStrategy(DEFECT),
              egt.behaviors.pgg_behaviors.PGGOneShotStrategy(COOPERATE)]



class gamePGG(egt.games.PGG):
    def __init__(self, Z, cost, M, strategies):
        super.__init__(Z, cost, M, strategies)
        
        
    def calculate_payoffs(self):
        matrix = []
        for s in range(self.nb_strategies_):
            payoffs = np.arra([])


game = egt.games.PGG(Z, cost, M, strategies)


print(game.calculate_payoffs())

state = np.array([150,50])

fit0 = game.calculate_fitness(DEFECT, Z, state)
fit1 = game.calculate_fitness(COOPERATE, Z, state)
print(fit0, fit1)