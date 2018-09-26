import numpy as np
from math import *

def enum(*sequential, **named):
	enums = dict(zip(sequential, range(len(sequential))), **named)
	return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'SIMULATION', 'DONE', 'STOP', 'CLOSED')

def calcEta(k, alpha, delta, maxSigma, n0):
	"""Calculate parameter eta.
	"""
	# return 3
	if k < 1000:
		return 3
	elif k < 10000:
		if n0 == 10:
			return 7
		elif n0 == 20:
			return 6
		else:
			return 5
	else:
		if n0 == 10:
			return 10
		elif n0 == 20:
			return 8
		else:
			return 6
		

def calcNs(k, eta, sigmas, delta):
	Ns = list(map(lambda sigma: int(np.ceil((2 * eta * sigma / delta) ** 2)), sigmas))
	return Ns

