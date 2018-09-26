from EP_test import EP
from genNormal import *

k = 10
alpha = 0.05
delta = 0.1
sigmas = [2] * k
n0 = 10
batchSize = 1

ep = EP(k, alpha, delta, sigmas, n0, batchSize, GenNormalSC)
ep.run()
