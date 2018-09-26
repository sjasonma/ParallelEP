from __future__ import division
from mpi4py import MPI
import time
from Util import *


# TODO:
#	3. Random stream management
#	4. calcEta

class Job:
	"""Job"""
	def __init__(self, sysid, n, seed, vecIdx=-1, lastOne=False):
		self.sysid = sysid
		self.n = n
		self.seed = seed
		self.vecIdx = vecIdx
		self.lastOne = lastOne
	def __str__(self):
		return "Job(%d, %d, %d, %d)" % (self.sysid, self.n, self.seed, 
			self.vecIdx)

class Result:
	"""Result"""
	def __init__(self, job, xsum, x2sum):
		self.sysid = job.sysid
		self.n = job.n
		self.vecIdx = job.vecIdx
		self.lastOne = job.lastOne
		self.xsum = xsum
		self.x2sum = x2sum
	def __str__(self):
		return "Result(%d, %d, %d: %.4f, %.4f)" % (self.sysid, self.n, 
			self.vecIdx, self.xsum, self.x2sum)

class VKN:
	""" Vector-Filling KN Procedure
	"""
	def __init__(self, simulator, alpha, delta, n0, batchSize, seed0, outputLevel):
		self.simulator = simulator
		self.k = simulator.getNumSystems()
		self.alpha = alpha
		self.delta = delta
		self.n0 = n0
		self.batchSize = batchSize
		self.seed = seed0

		self.comm = MPI.COMM_WORLD
		self.size = self.comm.size 
		self.rank = self.comm.rank
		self.status = MPI.Status()
		self.outputLevel = outputLevel

	def printPars(self):
		print("nCores=%d, k=%d, alpha=%.2f, delta=%.2f, n0=%d, bSize=%d, seed=%d" % (
			self.size, self.k, self.alpha, self.delta, self.n0, self.batchSize, self.seed))

	def getSeed(self):
		self.seed += 1
		return self.seed

	def generateJobs(self):
		for sysid in self.idx:
			self.jobList.append(Job(sysid, self.batchSize, 1))
		self.jobList[0].lastOne = True

	def sendJob(self, source):
		job = self.jobList.pop()
		job.vecIdx = self.jobCount
		self.jobCount += 1
		self.comm.send(job, dest=source, tag=tags.SIMULATION)
		if not self.jobList:
			self.generateJobs()
 
	def screening(self):
		idx_elim = np.full(self.idx.shape, False)
		for sysid in self.idx:
			S2 = np.var(self.x0s[sysid] - self.x0s[self.idx], 1) / (
				self.n0 - 1) * self.n0
			rW = np.maximum(0, self.h2 * S2 / (2 * self.delta) - self.delta 
				* self.r / 2)
			idx_elim = np.logical_or(idx_elim, self.xsums[self.idx] + 1e-8
				< self.xsums[sysid] - rW)
		self.idx = self.idx[~idx_elim]
		if self.outputLevel >= 1:
			if (self.r / self.batchSize) % 10 == 0:
				print("Round %d: %d systems left" % (self.r, len(self.idx)))

	def receiveResult(self, result):
		self.resultList[result.vecIdx % self.lenResultList] = result
		while self.resultList[self.resultPt] != 0:
			result = self.resultList[self.resultPt]
			sysid = result.sysid
			self.xsums[sysid] += result.xsum
			self.ns[sysid] += result.n
			if result.lastOne:
				self.r += self.batchSize
				self.screening()
			self.resultList[self.resultPt] = 0
			self.resultPt = (self.resultPt + 1) % self.lenResultList

	def master_initialize(self):
		self.num_workers = self.size - 1
		self.h2 = (self.n0 - 1) * (pow((2 * self.alpha / (self.k - 1)), (
			-2 / (self.n0 - 1))) - 1)
		self.x0s = np.empty([self.k, self.n0])
		self.workers_sim_time_0 = np.empty(self.num_workers)
		self.xsums = np.empty(self.k)
		self.ns = np.full(self.k, self.n0)
		self.idx = np.array(range(self.k))
		self.r = self.n0
		# Initialization
		print("Initialization")
		sysid = 0
		ni = 0
		closed_workers = 0
		while closed_workers < self.num_workers:
			data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, 
				status=self.status)
			source = self.status.Get_source()
			tag = self.status.Get_tag()
			if tag == tags.READY:
				if sysid < self.k:
					job = Job(sysid, 1, self.getSeed(), ni)
					self.comm.send(job, dest=source, tag=tags.SIMULATION)
					if self.outputLevel >= 3:
						print("Sending %s to worker %d" % (job, source))
					ni += 1
					if ni == self.n0:
						sysid += 1
						ni = 0
				else:
					self.comm.send(None, dest=source, tag=tags.STOP)
			elif tag == tags.DONE:
				result = data
				if self.outputLevel >= 3:
					print("Got %s from worker %d" % (result, source))
				self.x0s[result.sysid][result.vecIdx] = result.xsum
				if sysid < self.k:
					job = Job(sysid, 1, self.getSeed(), ni)
					self.comm.send(job, dest=source, tag=tags.SIMULATION)
					if self.outputLevel >= 3:
						print("Sending %s to worker %d" % (job, source))
					ni += 1
					if ni == self.n0:
						sysid += 1
						ni = 0
				else:
					self.comm.send(None, dest=source, tag=tags.STOP)
			elif tag == tags.CLOSED:
				if self.outputLevel >= 3:
					print("Worker %d exited." % source)
				closed_workers += 1
				self.workers_sim_time_0[source - 1] = data

		self.xsums = np.sum(self.x0s, 1)
		print("Initial screening")
		self.screening()
		print("Initial screening: %d systems left" % (len(self.idx)))

	def master_main(self):
		print("Working stage")
		closed_workers = 0
		isContinue = True
		self.lenResultList = 10 * self.num_workers
		self.jobCount = 0
		self.jobList = []
		self.resultList = [0] * self.lenResultList
		self.resultPt = 0
		self.workers_sim_time_1 = [-1] * self.num_workers

		self.generateJobs()
		while closed_workers < self.num_workers:
			if len(self.idx) == 1:
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
				if isContinue:
					self.sendJob(source)
				else:
					self.comm.send(None, dest=source, tag=tags.STOP)
				result = data
				self.receiveResult(result)
			elif tag == tags.CLOSED:
				closed_workers += 1
				self.workers_sim_time_1[source - 1] = data

		self.istar = self.idx[0]
		print("istar = %d, sum(n) = %d = %d + %d, n(istar) = %d" % (self.istar, sum(self.ns),
			self.n0 * self.k, sum(self.ns) - self.n0*  self.k, self.ns[self.istar]))


	def worker_execute(self):
		# Worker processes execute
		if self.outputLevel >= 3:
			print("Worker %d starts working!" % self.rank)
		time_sim_sum = 0
		self.comm.send(None, dest=0, tag=tags.READY)
		while True:
			job = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=self.status)
			tag = self.status.Get_tag()
			if tag == tags.SIMULATION:
				sim_start_time = time.time()
				xsum, x2sum = self.simulator.genObjective(job.sysid, job.n, 
					job.seed)
				time_sim_sum += time.time() - sim_start_time
				result = Result(job, xsum, x2sum)
				self.comm.send(result, dest=0, tag=tags.DONE)
			elif tag == tags.STOP:
				break
		self.comm.send(time_sim_sum, dest=0, tag=tags.CLOSED)
		if self.outputLevel >= 3:
			print("Worker %d ends working!" % self.rank)
			print("time for worker %d: %.2f" % (self.rank, time_sim_sum))

	def run(self):
		if self.rank == 0:
			self.printPars()
			start_time = time.time()
			self.master_initialize()
			total_wtime_0 = time.time() - start_time
			print("--- %s seconds ---" % total_wtime_0)
		else:
			self.worker_execute()
		self.comm.barrier()
		if self.rank == 0:
			self.master_main()
			total_wtime = time.time() - start_time
			print("--- %s seconds ---" % (total_wtime))
			total_sim_time = sum(self.workers_sim_time_0) + sum(self.workers_sim_time_1)
			utilization = total_sim_time / (total_wtime * self.num_workers)
			print("TotalSimTime = %.2f, Utilization = %.2f%%" % (total_sim_time, 
				utilization * 100))
		else:
			self.worker_execute()