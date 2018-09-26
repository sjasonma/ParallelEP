import numpy as np

class SimNormal:
	def __init__(self, mus, sigmas):
		self.mus = mus
		self.sigmas = sigmas
		self.k = len(mus)

	def getNumSystems(self):
		return self.k

	def genObjective(self, sysid, n, seed):
		Xs = np.random.normal(self.mus[sysid], self.sigmas[sysid], n)
		return (np.sum(Xs), np.sum(np.square(Xs)))

class SimNormalSC:
	def __init__(self, k, delta, sigma):
		mus = [0] * k
		mus[k-1] = delta
		sigmas = [sigma] * k
		SimNormal.__init__(self, mus, sigmas)

class SimNormalMIM:
	def __init__(self, k, delta, sigma):
		mus = [delta * x for x in xrange(k)]
		sigmas = [sigma] * k
		SimNormal.__init__(self, mus, sigmas)

class SimNormalRPI(SimNormal):
	def __init__(self, k, delta, sigma):
		mus = np.sort(np.random.normal(0, 2 * delta, k))
		sigmas = [sigma] * k
		SimNormal.__init__(self, mus, sigmas)
