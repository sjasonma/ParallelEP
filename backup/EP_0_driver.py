from EP_0 import EP
from TPMax import TpMax
from SimNormal import *

# mpiexec -np 4 python EP_0_driver.py

RB = 50
sim_tm = TpMax(RB)
k = sim_tm.getNumSystems()
alpha = 0.05
delta = 0.1
n0batch = 2
batchSize = 5
ep = EP(sim_tm, alpha, delta, n0batch, batchSize, 14850, 0)

# k = 10
# delta_mu = 0.1
# sigma = 2
# sim_rpi = SimNormalRPI(k, delta_mu, sigma)

# k = sim_rpi.getNumSystems()
# alpha = 0.05
# delta = 0.1
# n0batch = 2
# batchSize = 5
# ep = EP(sim_rpi, alpha, delta, n0batch, batchSize, 14850, 1)


ep.run()