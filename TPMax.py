from __future__ import division
import numpy as np
from math import *

class TpMax:
	def __init__(self, RB):
		self.RB = RB
		self.k = (RB - 1)* (RB - 2) // 2 * (RB - 1)

	def getNumSystems(self):
		return self.k

	def idx2sys(self, sysid):
		rr = self.RB * 2 - 3
		n = sysid // (self.RB - 1) + 1
		v0 = int(ceil((rr - sqrt(rr ** 2 - 8 * n)) / 2))
		v1= self.RB - 1 + (n - (rr-v0) * v0 // 2 ) - v0
		v2 = self.RB - v0 - v1;
		v3 = sysid % (self.RB - 1) + 1;
		v4 = self.RB - v3;
		v = (v0, v1, v2, v3, v4)
		return v

	def genObjective(self, sysid, n, seed):
		"""Return sum(Xi) and sum(Xi^2).
		"""
		sys = self.idx2sys(sysid)
		nstages = 3		# Number of stages
		warmup = 2000
		count = 50
		njobs = warmup + count
		r = sys[0: nstages]
		b = sys[nstages: ]
		tp = [0] * n
		sTimes = np.zeros((n, njobs, nstages))
		for j in range(nstages):
			sTimes[:, :, j] = np.random.exponential(1 / r[j], (n, njobs))
		for k in range(n):  
		    ETimes = np.zeros((nstages, njobs + 1))

		    for i in range(1, njobs + 1):
		        t = sTimes[k, i - 1, 0]
		        if ETimes[1, max(0, i - b[0])] <= ETimes[0, i - 1] + t:
		            ETimes[0, i] = ETimes[0, i - 1] + t
		        else:
		            ETimes[0, i] = ETimes[1, max(0, i- b[0])]

		        for j in range(1, nstages - 1):
		            t = sTimes[k, i - 1, j]
		            if ETimes[j, i - 1] < ETimes[j - 1, i]:
		                if ETimes[j + 1, max(0, i - b[j])] <= ETimes[j - 1, i] + t:
		                    ETimes[j, i] = ETimes[j - 1, i] + t
		                else:
		                    ETimes[j, i] = ETimes[j + 1, max(0, i - b[j])]
		            else:
		                if ETimes[j + 1, max(0,i - b[j])] <= ETimes[j, i - 1] + t:
		                    ETimes[j, i] = ETimes[j, i - 1] + t
		                else:
		                    ETimes[j, i] = ETimes[j + 1, max(0, i - b[j])]

		        t = sTimes[k, i - 1, nstages - 1]
		        if ETimes[nstages - 1, i - 1] <= ETimes[nstages - 2, i]:
		            ETimes[nstages - 1, i] = ETimes[nstages - 2, i] + t
		        else:
		            ETimes[nstages - 1, i] = ETimes[nstages - 1, i - 1] + t
		    
		    tp[k] = count / (ETimes[nstages - 1, njobs] - ETimes[nstages - 1, warmup])
		    
		FnSum, FnSumSq = np.sum(tp), np.sum(np.square(tp))
		return(FnSum, FnSumSq)

