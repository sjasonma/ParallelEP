import numpy as np

class GenNormal:
	def __init__(self, k, mus, sigmas):
		self.k = k
		self.mus = mus
		self.sigmas = sigmas
	def run(self, sysid, n):
		return sum(np.random.normal(self.mus[sysid], self.sigmas[sysid], n))

class GenNormalSC:
	def __init__(self, k, delta, sigmas):
		mus = [0] * k
		mus[k-1] = delta
		self.generator = GenNormal(k, mus, sigmas)
	def run(self, sysid, n):
		return self.generator.run(sysid, n)

class GenNormalMIM:
	def __init__(self, k, delta, sigmas):
		mus = [delta * x for x in xrange(k)]
		self.generator = GenNormal(k, mus, sigmas)
	def run(self, sysid, n):
		return self.generator.run(sysid, n)

