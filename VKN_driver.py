from VKN import VKN
from TPMax import TpMax
from SimNormal import *

# mpiexec -np 4 python VKN_driver.py

# RB = 10
# sim_tm = TpMax(RB)
# k = sim_tm.getNumSystems()
# alpha = 0.05
# delta = 0.1
# n0 = 10
# batchSize = 10
# vkn = VKN(sim_tm, alpha, delta, n0, batchSize, 14850, 1)

RB = 50
sim_tm = TpMax(RB)
k = sim_tm.getNumSystems()
alpha = 0.05
delta = 0.1
n0 = 50
batchSize = 50
vkn = VKN(sim_tm, alpha, delta, n0, batchSize, 14850, 1)

# k = 10
# delta_mu = 0.1
# sigma = 2
# sim_rpi = SimNormalRPI(k, delta_mu, sigma)
# k = sim_rpi.getNumSystems()
# alpha = 0.05
# delta = 0.1
# n0 = 10
# batchSize = 5
# vkn = VKN(sim_rpi, alpha, delta, n0, batchSize, 14850, 1)


vkn.run()

