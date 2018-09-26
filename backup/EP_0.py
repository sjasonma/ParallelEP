from mpi4py import MPI
import time
from Util import *


# TODO:
#	3. Random stream management
#	4. calcEta

class Job:
	"""Job"""
	def __init__(self, sysid, n, seed, vecIdx=-1):
		self.sysid = sysid
		self.n = n
		self.seed = seed
		self.vecIdx = vecIdx
	def __str__(self):
		return "Job(%d, %d, %d, %d)" % (self.sysid, self.n, self.seed, 
			self.vecIdx)

class Result:
	"""Result"""
	def __init__(self, job, xsum, x2sum):
		self.sysid = job.sysid
		self.n = job.n
		self.vecIdx = job.vecIdx
		self.xsum = xsum
		self.x2sum = x2sum
	def __str__(self):
		return "Result(%d, %d, %d: %.4f, %.4f)" % (self.sysid, self.n, 
			self.vecIdx, self.xsum, self.x2sum)


class EP:
	"""Parallel envelope procedure in master-worker scheme
	"""
	def __init__(self, simulator, alpha, delta, n0batch, batchSize, seed0, outputLevel):
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
		self.outputLevel = outputLevel

	def getSeed(self):
		self.seed += 1
		return self.seed

	def updateU(self, sysid):
		self.Us[sysid] = self.xbars[sysid] + self.eta * self.sigmas[sysid] / (
			self.ns[sysid] + self.ms[sysid]) ** 0.5
				
	def checkStoppingRule(self):
		istar= self.istar
		L_istar = (self.xbars[istar] - self.eta * 
			self.sigmas[istar] / self.ns[istar]**0.5)
		U_jstar = self.Us[self.jstar]
		return L_istar > U_jstar - self.delta

	def generateJobs(self):
		""" Two-sample rule
		"""
		self.jobList = [Job(self.jstar, self.batchSize, 1, -1),
		Job(self.istar, self.batchSize, 1, -1)]

	def sendJob(self, source):
		job = self.jobList.pop()
		job.n = min(job.n, self.Ns[job.sysid] - self.ns[job.sysid] 
			- self.ms[job.sysid])
		job.vecIdx = self.jobCount
		self.jobCount += 1
		self.comm.send(job, dest=source, tag=tags.SIMULATION)
		self.ms[job.sysid] += job.n
		self.updateU(job.sysid)
		self.jstar = np.argmax(self.Us[: self.istar] + self.Us[self.istar + 1:])
		if self.jstar >= self.istar:
			self.jstar += 1
		if not self.jobList:
			self.generateJobs()

	def receiveResult(self, result):
		istarOld = self.istar
		self.resultList[result.vecIdx % self.lenResultList] = result
		while self.resultList[self.resultPt] != 0:
			result = self.resultList[self.resultPt]
			sysid = result.sysid
			self.xbars[sysid] = (self.xbars[sysid] * self.ns[sysid] + 
				result.xsum) / (self.ns[sysid] + result.n)
			self.ns[sysid] += result.n
			self.ms[sysid] -= result.n
			self.updateU(sysid)
			self.resultList[self.resultPt] = 0
			self.resultPt = (self.resultPt + 1) % self.lenResultList
		self.istar = np.argmax(self.xbars)
		self.jstar = np.argmax(self.Us[: self.istar] + self.Us[self.istar + 1:])
		if self.jstar >= self.istar:
			self.jstar += 1
		if self.istar != istarOld:
			self.generateJobs()

	def master_initialize(self):

		self.num_workers = self.size - 1
		print("Master starting with %d workers" % self.num_workers)
		self.ns = [self.n0batch * self.batchSize] * self.k
		self.ms = [0] * self.k
		self.xsums = [0] * self.k
		self.x2sums = [0] * self.k
		
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
					job = Job(sysid, self.batchSize, self.getSeed())
					self.comm.send(job, dest=source, tag=tags.SIMULATION)
					if self.outputLevel >= 2:
						print("Sending %s to worker %d" % (job, source))
					batch += 1
					if batch == self.n0batch:
						batch = 0
						sysid += 1
				else:
					self.comm.send(None, dest=source, tag=tags.STOP)
			elif tag == tags.DONE:
				result = data
				if self.outputLevel >= 2:
					print("Got %s from worker %d" % (result, source))
				self.xsums[result.sysid] += result.xsum
				self.x2sums[result.sysid] += result.x2sum
			elif tag == tags.CLOSED:
				if self.outputLevel >= 2:
					print("Worker %d exited." % source)
				closed_workers += 1

		self.xbars = [xsum / n for xsum, n in zip(self.xsums, self.ns)]
		self.sigmas = [sqrt((x2sum - (xsum) ** 2 / n )/ (n - 1)) for xsum, 
			x2sum, n in zip(self.xsums, self.x2sums, self.ns)]
		self.eta = calcEta(self.k, self.alpha, self.delta, max(self.sigmas))
		self.Ns = calcNs(self.k, self.eta, self.sigmas, self.delta)
		self.Us = [xsum / n + self.eta * sigma / n**0.5 for xsum, sigma, n 
		in zip(self.xsums, self.sigmas, self.ns)]

		self.istar = np.argmax(self.xbars)
		self.jstar = np.argmax(self.Us[: self.istar] + self.Us[self.istar + 1:])
		if self.jstar >= self.istar:
			self.jstar += 1

		if self.outputLevel >= 1:
			for i in range(self.k):
				print("%d: %.4f, %.4f" % (i, self.xbars[i], self.Us[i]))
		print("istar = %d, jstar = %d" % (self.istar, self.jstar))



	def master_main(self):
		print("Working stage")
		closed_workers = 0
		isContinue = True
		self.lenResultList = 10 * self.num_workers
		self.jobCount = 0
		self.jobList = []
		self.resultList = [0] * self.lenResultList
		self.resultPt = 0

		self.generateJobs()
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

		if self.outputLevel >= 1:
			for i in range(self.k):
				print("%d: %.4f, %d, %.4f" % (i, self.xbars[i], 
					self.ns[i], self.Us[i]))
		print("istar = %d, sum(n) = %d" % (self.istar, sum(self.ns)))


	def worker_execute(self):
		# Worker processes execute
		while True:
			self.comm.send(None, dest=0, tag=tags.READY)
			job = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=self.status)
			tag = self.status.Get_tag()
			if tag == tags.SIMULATION:
				xsum, x2sum = self.simulator.genObjective(job.sysid, job.n, 
					job.seed)
				result = Result(job, xsum, x2sum)
				self.comm.send(result, dest=0, tag=tags.DONE)
			elif tag == tags.STOP:
				break
		self.comm.send(None, dest=0, tag=tags.CLOSED)


	def run(self):
		if self.rank == 0:
			start_time = time.time()
			self.master_initialize()
			print("--- %s seconds ---" % (time.time() - start_time))
			self.master_main()
			print("--- %s seconds ---" % (time.time() - start_time))
		else:
			self.worker_execute()
			self.worker_execute()