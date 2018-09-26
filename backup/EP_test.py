import numpy as np
from math import *
from mpi4py import MPI

# 2-sample strategy
# TODO: Add vector-filling
# TODO: Use heap for i* and j*?

def enum(*sequential, **named):
	enums = dict(zip(sequential, range(len(sequential))), **named)
	return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START')

def calcEta(k, alpha, delta, maxSigma):
	"""Calculate parameter eta.
	"""
	return 3

def calcNs(k, eta, sigmas, delta):
	Ns = map(lambda sigma: int(ceil((2 * eta * sigma / delta) ** 2)), sigmas)
	return Ns

class Job:
	"""Job"""
	def __init__(self, sysid, n, vecIdx):
		self.sysid = sysid
		self.n = n
		self.vecIdx = vecIdx
	def __str__(self):
		return "Job(%d, %d, %d)" % (self.sysid, self.n, self.vecIdx)

class Result:
	"""Result"""
	def __init__(self, job, Xsum):
		self.sysid = job.sysid
		self.vecIdx = job.vecIdx
		self.n = job.n
		self.Xsum = Xsum
	def __str__(self):
		return "Result(%d, %d, %d: %.4f)" % (self.sysid, self.n, self.vecIdx, self.Xsum)


class EP:
	"""Parallel envelope procedure in master-worker scheme
	"""
	def __init__(self, k, alpha, delta, sigmas, n0, batchSize, simulator):
		self.k = k
		self.alpha = alpha
		self.delta = delta
		self.sigmas = sigmas
		self.n0 = n0
		self.batchSize = batchSize
		self.simulator = simulator(k, delta, sigmas)

		self.comm = MPI.COMM_WORLD
		self.size = self.comm.size 
		self.rank = self.comm.rank
		self.status = MPI.Status()

	def calcU(self, sysid):
		U = self.Xsums[sysid] / self.ns[sysid] + self.eta * self.sigmas[sysid] / (self.ns[sysid]
			+ self.ms[sysid])**0.5
		return U

	def updateIstarJstarFull(self):
		ix, ju1, ju2 = float('-inf'), float('-inf'), float('-inf')
		jstar1 = -1
		for i in xrange(self.k):
			if self.Xsums[i] / self.ns[i] > ix:
				ix, istar = self.Xsums[i] / self.ns[i], i
			if self.Us[i] > ju2:
				if self.Us[i] > ju1:
					jstar1, ju1, jstar2, ju2 = i, self.Us[i], jstar1, ju1
				else:
					jstar2, ju2 = i, self.Us[i]
		if istar == jstar1:
			self.istar, self.jstar = istar, jstar2
		else:
			self.istar, self.jstar = istar, jstar1
				
	def checkStoppingRule(self):
		istar= self.istar
		L_istar = self.Xsums[istar] / self.ns[istar] - self.eta * self.sigmas[istar] / self.ns[istar]**0.5
		U_jstar = self.Us[self.jstar]
		return L_istar > U_jstar - self.delta

	def samplingRule(self):
		"""
		2-Sample Strategy
		"""
		if self.jobList:
			job = self.jobList.pop()
			return job
		else:
			self.jobList = [Job(self.jstar, self.batchSize, 1),
			Job(self.istar, self.batchSize, 1)]
			job = self.jobList.pop()
			return job

	def updateStatsSend(self, sysid):
		self.Us[sysid] = self.calcU(sysid)
		self.updateIstarJstarFull()

	def updateStatsReceive(self, result):
		sysid = result.sysid
		self.Xsums[sysid] += result.Xsum
		self.ns[sysid] += result.n
		self.ms[sysid] -= result.n
		self.Us[sysid] = self.calcU(sysid)
		self.updateIstarJstarFull()

	def master_initialize(self):

		self.num_workers = self.size - 1
		print("Master starting with %d workers" % self.num_workers)
		self.eta = calcEta(self.k, self.alpha, self.delta, max(self.sigmas))
		self.Ns = calcNs(self.k, self.eta, self.sigmas, self.delta)
		self.ns = [self.n0] * self.k
		self.ms = [0] * self.k
		self.Xsums = [0] * self.k
		self.jobList = []
		
		# Initialization
		print("Initialization")
		n0Batches = self.n0 / self.batchSize
		sysid = 0
		batch = 0
		closed_workers = 0
		while closed_workers < self.num_workers:
			data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=self.status)
			source = self.status.Get_source()
			tag = self.status.Get_tag()
			if tag == tags.READY:
				if sysid < self.k:
					job = Job(sysid, self.batchSize, batch)
					self.comm.send(job, dest=source, tag=tags.START)
					#print("Sending %s to worker %d" % (job, source))
					batch += 1
					if batch == n0Batches:
						batch = 0
						sysid += 1
				else:
					self.comm.send(None, dest=source, tag=tags.EXIT)
			elif tag == tags.DONE:
				result = data
				#print("Got %s from worker %d" % (result, source))
				self.Xsums[result.sysid] += result.Xsum
			elif tag == tags.EXIT:
				#print("Worker %d exited." % source)
				closed_workers += 1
		self.Us = [Xsum / n + self.eta * sigma / n**0.5 for Xsum, sigma, n in
		zip(self.Xsums, self.sigmas, self.ns)]
		self.updateIstarJstarFull()

		# for i in xrange(self.k):
		# 	print("%d: %.4f, %.4f" % (i, self.Xsums[i]/self.ns[i], self.Us[i]))


	def master_main(self):
		print("Working stage")
		closed_workers = 0
		isContinue = True
		while closed_workers < self.num_workers:
			if self.checkStoppingRule():
				isContinue = False
			data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=self.status)
			source = self.status.Get_source()
			tag = self.status.Get_tag()
			if tag == tags.READY:
				if isContinue:
					job = self.samplingRule()
					job.n = min(job.n, self.Ns[job.sysid] - self.ns[job.sysid] - self.ms[job.sysid])
					self.comm.send(job, dest=source, tag=tags.START)
					self.ms[job.sysid] += job.n
					self.updateStatsSend(job.sysid)
				else:
					self.comm.send(None, dest=source, tag=tags.EXIT)
			elif tag == tags.DONE:
				result = data
				self.updateStatsReceive(result)
			elif tag == tags.EXIT:
				closed_workers += 1
		print("Master finishing")

		for i in xrange(self.k):
			print("%d: %.4f, %d, %.4f" % (i, self.Xsums[i]/self.ns[i], self.ns[i], self.Us[i]))
		print("istar = %d, sum(n) = %d" % (self.istar, sum(self.ns)))

	def worker_execute(self):
		# Worker processes execute code below
		# Initialization
		while True:
			self.comm.send(None, dest=0, tag=tags.READY)
			job = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=self.status)
			tag = self.status.Get_tag()
			if tag == tags.START:
				Xsum = self.simulator.run(job.sysid, job.n)
				result = Result(job, Xsum)
				self.comm.send(result, dest=0, tag=tags.DONE)
			elif tag == tags.EXIT:
				break
		self.comm.send(None, dest=0, tag=tags.EXIT)
		# Stopping rule and sampling rule

	def run(self):
		if self.rank == 0:
			self.master_initialize()
			self.master_main()
		else:
			self.worker_execute()
			self.worker_execute()