from mpi4py import MPI
from Util import *

# TODO:
#	1. Vector filling
#	2. Sampling rule
#	3. Random stream management
#	4. calcEta
#	5. 

class Job:
	"""Job"""
	def __init__(self, sysid, n, vecIdx, seed):
		self.sysid = sysid
		self.n = n
		self.vecIdx = vecIdx
		self.seed = seed
	def __str__(self):
		return "Job(%d, %d, %d, %d)" % (self.sysid, self.n, self.seed, 
			self.vecIdx)

class Result:
	"""Result"""
	def __init__(self, job, Xsum, X2sum):
		self.sysid = job.sysid
		self.n = job.n
		self.vecIdx = job.vecIdx
		self.Xsum = Xsum
		self.X2sum = X2sum
	def __str__(self):
		return "Result(%d, %d, %d: %.4f, %.4f)" % (self.sysid, self.n, 
			self.vecIdx, self.Xsum, self.X2sum)


class EP:
	"""Parallel envelope procedure in master-worker scheme
	"""
	def __init__(self, simulator, alpha, delta, n0batch, batchSize, seed0):
		self.simulator = simulator
		self.k = simulator.getNumSystems()
		self.alpha = alpha
		self.delta = delta
		self.n0batch = n0batch
		self.batchSize = batchSize
		self.seed = seed0

		self.comm = MPI.COMM_WORLD
		self.size = self.comm.size 
		self.rank = self.comm.rank
		self.status = MPI.Status()

	def getSeed(self):
		self.seed += 1
		return self.seed

	def calcU(self, sysid):
		U = self.Xsums[sysid] / self.ns[sysid] + self.eta * self.sigmas[sysid] / (
			self.ns[sysid] + self.ms[sysid]) ** 0.5
		return U

	def updateIstarJstarFull(self):
		ix, ju1, ju2 = float('-inf'), float('-inf'), float('-inf')
		jstar1 = -1
		for i in range(self.k):
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
		L_istar = (self.Xsums[istar] / self.ns[istar] - self.eta * 
			self.sigmas[istar] / self.ns[istar]**0.5)
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
			self.jobList = [Job(self.jstar, self.batchSize, 1, 1),
			Job(self.istar, self.batchSize, 1, 1)]
			job = self.jobList.pop()
			return job

	def updateStatsSend(self, sysid):
		self.Us[sysid] = self.calcU(sysid)
		self.updateIstarJstarFull()

	def sendJob(self, source):
		job = self.samplingRule()
		job.n = min(job.n, self.Ns[job.sysid] - self.ns[job.sysid] 
			- self.ms[job.sysid])
		self.comm.send(job, dest=source, tag=tags.SIMULATION)
		self.ms[job.sysid] += job.n
		self.updateStatsSend(job.sysid)

	def receiveResult(self, result):
		sysid = result.sysid
		self.Xsums[sysid] += result.Xsum
		self.ns[sysid] += result.n
		self.ms[sysid] -= result.n
		self.Us[sysid] = self.calcU(sysid)
		self.updateIstarJstarFull()

	def master_initialize(self):

		self.num_workers = self.size - 1
		print("Master starting with %d workers" % self.num_workers)
		self.ns = [self.n0batch * self.batchSize] * self.k
		self.ms = [0] * self.k
		self.Xsums = [0] * self.k
		self.X2sums = [0] * self.k
		self.jobList = []
		
		# Initialization
		print("Initialization")
		sysid = 0
		batch = 0
		closed_workers = 0
		while closed_workers < self.num_workers:
			data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, 
				status=self.status)
			source = self.status.Get_source()
			tag = self.status.Get_tag()
			if tag == tags.READY:
				if sysid < self.k:
					job = Job(sysid, self.batchSize, batch, self.getSeed())
					self.comm.send(job, dest=source, tag=tags.SIMULATION)
					#print("Sending %s to worker %d" % (job, source))
					batch += 1
					if batch == self.n0batch:
						batch = 0
						sysid += 1
				else:
					self.comm.send(None, dest=source, tag=tags.STOP)
			elif tag == tags.DONE:
				result = data
				#print("Got %s from worker %d" % (result, source))
				self.Xsums[result.sysid] += result.Xsum
				self.X2sums[result.sysid] += result.X2sum
			elif tag == tags.CLOSED:
				#print("Worker %d exited." % source)
				closed_workers += 1

		self.sigmas = [sqrt((X2sum - (Xsum) ** 2 / n )/ (n - 1)) for Xsum, 
			X2sum, n in zip(self.Xsums, self.X2sums, self.ns)]
		self.eta = calcEta(self.k, self.alpha, self.delta, max(self.sigmas))
		self.Ns = calcNs(self.k, self.eta, self.sigmas, self.delta)
		self.Us = [Xsum / n + self.eta * sigma / n**0.5 for Xsum, sigma, n 
		in zip(self.Xsums, self.sigmas, self.ns)]

		# for i in range(self.k):
		# 	print("%d: %.4f, %.4f" % (i, self.Xsums[i]/self.ns[i], self.Us[i]))


	def master_main(self):
		print("Working stage")
		closed_workers = 0
		isContinue = True
		self.updateIstarJstarFull()
		while closed_workers < self.num_workers:
			if isContinue and self.checkStoppingRule():
				isContinue = False
			data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, 
				status=self.status)
			source = self.status.Get_source()
			tag = self.status.Get_tag()
			if tag == tags.READY:
				if isContinue:
					self.sendJob(source)
				else:
					self.comm.send(None, dest=source, tag=tags.STOP)
			elif tag == tags.DONE:
				result = data
				self.receiveResult(result)
			elif tag == tags.CLOSED:
				closed_workers += 1
		print("Master finishing")

		for i in range(self.k):
			print("%d: %.4f, %d, %.4f" % (i, self.Xsums[i]/self.ns[i], 
				self.ns[i], self.Us[i]))
		print("istar = %d, sum(n) = %d" % (self.istar, sum(self.ns)))


	def worker_execute(self):
		# Worker processes execute
		while True:
			self.comm.send(None, dest=0, tag=tags.READY)
			job = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=self.status)
			tag = self.status.Get_tag()
			if tag == tags.SIMULATION:
				Xsum, X2sum = self.simulator.genObjective(job.sysid, job.n, 
					job.seed)
				result = Result(job, Xsum, X2sum)
				self.comm.send(result, dest=0, tag=tags.DONE)
			elif tag == tags.STOP:
				break
		self.comm.send(None, dest=0, tag=tags.CLOSED)


	def run(self):
		if self.rank == 0:
			self.master_initialize()
			self.master_main()
		else:
			self.worker_execute()
			self.worker_execute()