import numpy as np
import scipy
import sklearn
import json
import networkx as nx
import sys
import io
import threading
import time
from l1regls import l1regls
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from cvxopt import matrix
from networkx.readwrite import json_graph


def syntheticGenerator(n,d,N,sigma):
	"""
	The data generated by this function may have non-independent subspaces
	n 		: number of feature
	d 		: a list of dimension for each subspace
	N 		: a list of sample size for each subspace
	sigma 	: noise-level (i.e.  Y = X + E, E ~ Normal(0,sigma^2))
	"""
	X = []
	label = []
	if len(d) != len(N) or len(d) == 0:
		return None
	for subspace in xrange(len(N)):
		dim = d[subspace]
		Num = N[subspace]
		if dim > n:
			return None 
		base = np.random.uniform(0,1,(dim,n))
		base = sklearn.preprocessing.normalize(base,axis=1)
		X.extend((np.random.dirichlet(np.ones(dim),Num)).dot(base) + np.random.normal(0,sigma/np.sqrt(n),(Num,n)))
		label.extend([subspace for i in xrange(Num)])
	mapping = np.array([np.array([i,label[i]]) for i in xrange(len(label))])
	mapping = mapping[np.random.permutation(len(mapping))]
	return np.array(X)[mapping[:,0]],mapping[:,1]
"""
(e.g.)
from test import syntheticGenerator as sg
X,y = sg(5,[1,2],[4,5],0.05)
X,y = sg(10,[2,3,3,4],[1000,3000,3000,7000],0.05)
"""
def parseCMUMotionData(filename):
	"""
	In this experiments, we use the CMU Motion Data
	http:// mocap.cs.cmu.edu
	subject86, trial{2,5}
	this function would return data X with dimension M * N, 
	where M is the sample size and N is the number of feature
	"""
	X = []
	with open(filename,'r') as f:
		f.readline()
		f.readline()
		f.readline()
		buf = f.readline()
		while(buf != ""):
			print buf.replace('\n','')
			x=[]
			for i in xrange(0,29):
				buf = f.readline()
				x.extend([float(s) for s in buf.replace('\n','').split(' ')[1:]])
			X.append(x)
			buf = f.readline()
	return X

def falseDiscovery(y,ci,i):
	"""
	Given a sparse representation ci of instance i, and true clustering labels y,
	count the number of false discovery and the number of non-zero component of ci.
	"""
	num = 0
	fd = 0
	for j in xrange(len(ci)):
		if ci[j] == 0:
			continue
		num = num + 1
		if y[i] != y[j]:
			fd = fd + 1
	return fd,num

def solveL1NormWithRelaxation(A,b):
	"""
		minimize ||x||_1 + ||Ax - b||_2^2
			x
	"""
	return l1regls(A,b)

def solveL1NormExactly(A,b):
	""" 
		minimize  ||x||_1
			x
		s.t Ax = b

	It does not always have a feasible solution, that's why we need some relaxation
	"""
	from scipy.optimize import linprog
	X = np.array(A)
	O = np.zeros_like(X)
	I = np.identity(X.shape[1])
	G = np.vstack((np.vstack((np.hstack((X,O)),np.hstack((-X,O)))),np.vstack((np.hstack((I,-I)),np.hstack((-I,-I))))))
	c = np.hstack((np.zeros((X.shape[1])),np.ones((X.shape[1]))))
	h = np.hstack((b,-b,np.zeros((X.shape[1]*2))))
	return linprog(c,G,h)

def constructSR(X,zeroThreshold=1e-7,aprxInf=9e+4):
	"""
	First solve 

		minimize ||c||_1 + aprxInf^2 * ||c^T X_{-n} - x_n||_2^2
			c

	which is an approximation of

		minimize ||c||_1
			c
		s.t  c^T X_{-n} = x_n

	provided that aprxInf is sufficiently large.
	Let c* denotes the solution to the above problem.
	Since (Mahdi Soltanolkotabi et al. Robust Subspace Clustering, 2013) found that
	better choice of lambda is around 1/sqrt(dimension),
	where 
			sqrt(dimension) ~= 0.25 / ||c*||_1

	Therefore, we formulate the second-phase optimization problem as

		minimize ||c_1|| + (4*||c*||_1 / 2)||c^T X_{-n} - x_n||_2
			c
	
	This function will find the Sparse Representation for each instance sequentially,
	and then return C = [c_1, c_2, c_3, ...]
	"""
	C = np.zeros((len(X),len(X)))
	for n in xrange(len(X)):
		A = X
		A = np.delete(A,n,axis=0)
		w = solveL1NormWithRelaxation(matrix(A).T*aprxInf,matrix(X[n])*aprxInf)# Approximate to the l1-norm minimization with equality constraint
		lambd = np.sqrt(2*np.sum(np.abs(np.array(w))))  # Estimate the dimension of subspace and find a proper lambda
		w = solveL1NormWithRelaxation(matrix(A).T*lambd,matrix(X[n])*lambd)
		for i in xrange(n):
			if abs(w[i]) > zeroThreshold:
				C[n][i] = w[i]
		for i in xrange(n,len(w)):
			if abs(w[i]) > zeroThreshold:
				C[n][i+1] = w[i]
		print n,(2*lambd**2)**2  #print index and dimension of subspace
	return C.tolist()
def constructAffinityGraph(C):
	"""
	Given Sparse Representation matrix,
	construct an networkx.Graph() opbject
	"""
	G = nx.Graph()
	for i in xrange(len(C)):
		for j in xrange(i+1,len(C)):
			G.add_edge(i,j,weight=abs(C[i][j])+abs(C[j][i]))
	return G

def spectralClustering(G):
	"""
	perform spectral clustering on the laplacian of Sparse Representation.
	The input is the networkx.Graph() object constructed by Sparse Representation.
	This function will return each subspace label for each instance in the form of list.
	"""
	L = nx.laplacian_matrix(G).todense()
	w,v = scipy.linalg.eig(L)
	v = v.T
	w = sorted([(w[i],i) for i in xrange(len(w))],key=lambda x : x[0])
	maxgap = 0
	idx = 0
	for i in xrange(1,len(w)):
		if maxgap < w[i][0] - w[i-1][0]:
			maxgap = w[i][0] - w[i-1][0]
			idx = i
	
	v = [v[w[i][1]] for i in xrange(idx)]
	km = KMeans(idx)
	result = km.fit_predict(np.array(v).T)
	return result.tolist()

def fastSSC(X,filename="",numThreads=16,zeroThreshold=1e-7,aprxInf=9e+4,write=False):
	"""
	perform
		constructSR() -> constructAffinityGraph() -> spectralClustering()
	in a multi-threading way.
	"""
	C = np.zeros((len(X),len(X)))
	class SRcalculator(threading.Thread):
		def __init__(self, idxList, numTotal):
			threading.Thread.__init__(self)
			self.idxList = idxList
			self.tC = np.zeros((len(idxList),numTotal))
		def run(self):
			calculateL1regLS(self.tC,self.idxList)

	def calculateL1regLS(tC,idxList):
		for idx in xrange(len(idxList)):
			n = idxList[idx]
			A = X
			A = np.delete(A,n,axis=0)
			w = solveL1NormWithRelaxation(matrix(A).T*aprxInf,matrix(X[n])*aprxInf)# Approximate to the l1-norm minimization with equality constraint
			lambd = np.sqrt(2*np.sum(np.abs(np.array(w))))  # Estimate the dimension of subspace and find a proper lambda
			w = solveL1NormWithRelaxation(matrix(A).T*lambd,matrix(X[n])*lambd)
			for i in xrange(n):
				if abs(w[i]) > zeroThreshold:
					tC[idx][i] = w[i]
				else:tC[idx][i] = 0
			for i in xrange(n,len(w)):
				if abs(w[i]) > zeroThreshold:
					tC[idx][i+1] = w[i]
				else:tC[idx][i+1] = 0
		for idx in xrange(len(idxList)):
			n = idxList[idx]
			for j in xrange(len(C[n])):
				C[n][j] = tC[idx][j]
	Threads = []
	for i in xrange(numThreads):
		if i == numThreads-1:
			Threads.append(SRcalculator(range(i*(len(X)/numThreads),len(X)),len(X)))
		else: Threads.append(SRcalculator(range(i*(len(X)/numThreads),(i+1)*(len(X)/numThreads)),len(X)))
	for t in Threads:
		t.start()
	for t in Threads:
		t.join()
	if write and filename != "":
		with open("SR_"+filename,'w+') as f:
			json.dump(C.tolist(),f)
	result = spectralClustering(constructAffinityGraph(C))
	if write and filename != "":
		with open("result_"+filename,'w+') as f:
			json.dump(result,f)
	return result

def sparseSubspaceClustering(X,filename="",numThreads=1,zeroThreshold=1e-7,aprxInf=9e+4):
	"""
	The original sparse subspace clustering proposed in 
	(Ehsan Elhamifar, et al. Sparse Subspace Clustering, 2009)
	which consists of three steps:
		I. 		find sparse representation for each instance
		II. 	construct affinity matrix based on the sparse representation
		III. 	apply spectral clustering on the affinity matrix
	"""
	if numThreads > 1:
		return fastSSC(X,filename,numThreads=16,zeroThreshold=1e-7,aprxInf=9e+4,write=True)
	else:
		C = constructSR(X,zeroThreshold,aprxInf)
		if filename != "":
			with open("SR_"+filename,'w+') as f:
				json.dump(C,f)
		result = spectralClustering(constructAffinityGraph(C))
		if filename != "":
			with open("result_"+filename,'w+') as f:
				json.dump(result,f)
		return result


def subSampling(S,T=set()):
	"""
	Construct subsample according to 
	(Steve Hanneke. The Optimal Sample Complexity of PAC Learning, 2016)
	"""
	S = set(S)
	if len(S) <= 3:
		return [list(S.union(T))]
	size = len(S)/4
	S0 = set(list(S)[:-3*size])
	S1 = set(list(S)[-2*size:-size]).union(list(S)[-size:]).union(T)
	S2 = set(list(S)[-3*size:-2*size]).union(list(S)[-size:]).union(T)
	S3 = set(list(S)[-3*size:-2*size]).union(list(S)[-2*size:-size]).union(T)
	R = subSampling(S0,S1)
	R.extend(subSampling(S0,S2))
	R.extend(subSampling(S0,S3))
	return R

def ensembleSparseSubspaceClustering(X,filename="",numThreads=16,zeroThreshold=1e-7,aprxInf=9e+4):
	"""
	The method proposed in our work.
	First construct subsamples according to 
	(Steve Hanneke. The Optimal Sample Complexity of PAC Learning, 2016),
	then apply Robust Subspace Clustering on each subsample to get a set of clustering results.
	Base on these results to construct the final graph.
	We use majority voting to decide whether there should be an edge between each pair of instances.
	Finally, apply spectral clustering on the final graph.
	"""
	subSamples = subSampling(range(len(X)))
	A = dict()
	numSubSample = len(subSamples)
	for subsample in subSamples:
		if numThreads > 1:
			result = fastSSC(X[subsample],filename,numThreads=16,zeroThreshold=1e-7,aprxInf=9e+4,write=False)
		else: result = spectralClustering(constructAffinityGraph(constructSR(X[subsample],zeroThreshold,aprxInf)))
		for i in xrange(len(result)):
			for j in xrange(i+1,len(result)):
				if result[i] == result[j]:
					if (i,j) not in A:
						A[i,j] = 1
					else: A[i][j] = A[i][j] + 1
	G = nx.Graph()
	for edge in A:
		if A[edge]*2 > numSubSample:  		#majority voting
			G.add_edge(edge[0],edge[1])
	with open("Ensemble_graph_"+filename,'w+') as f:
		json.dump(json_graph.node_link_data(G),f)
	result = spectralClustering(G)
	with open("Ensemble_result_"+filename,'w+') as f:
		json.dump(result,f)
	return result

def evaluate(a,b):
	"""
	Given two list of clustering result, count the number of consistent pair of instances
	It is called consistent if (a[i] == a[j] and b[i] == b[j]) or (a[i] != a[j] and b[i] != b[j])
	where i,j are different instances and a[i] denotes the label of subspace to which instance i belongs
	"""
	num = 0
	error = 0
	for i in xrange(len(a)):
		for j in xrange(i+1,len(a)):
			num = num + 1
			if (a[i]==a[j] and b[i]!=b[j]) or (a[i]!=a[j] and b[i]==b[j]):
				error = error + 1
	return error,num
def countNonzero(v,zeroThreshold=1e-7):
	"""
	Count the number of non-zero components of the vector v
	"""
	s = 0
	for e in v:
		if abs(e) > zeroThreshold:
			s = s + 1
	return s

if __name__ == "__main__":
	
	file = sys.argv[1]
	dire = sys.argv[2]
	rbegin = int(sys.argv[3])
	rend = int(sys.argv[4])
	# X = parseCMUMotionData(file)
	with open(file,'r') as f:
		X = json.load(f)
	y = X[1]
	X = X[0]

	X = np.array(X)
	X = normalize(X,axis=1)
	subSamples = subSampling(range(len(X)))
	if rend == -1:
		rend = len(subSamples)
	if rend > len(subSamples):
		print len(subSamples)
		rend = rbegin
	for i in xrange(rbegin,rend):
		subsample = subSamples[i]
		print i,rend,len(subSamples),len(subsample)
		# Y = np.array([X[j] for j in subsample])
		C = constructSR(X[subsample],zeroThreshold=1e-7,aprxInf=9e+4)
		with open(dire+str(i),'w+') as f:
			json.dump(C,f)

	# filename = sys.argv[1]
	# X = parseCMUMotionData(filename)
	# X = np.array(X)
	# X = normalize(X,axis=1)
	# ensembleSparseSubspaceClustering(X,filename,numThreads=1,zeroThreshold=1e-7,aprxInf=9e+4)
	# sparseSubspaceClustering(X,filename,numThreads=1,zeroThreshold=1e-7,aprxInf=9e+4)
	
	"""Synthetic 1
	X,y = syntheticGenerator(n=100,d=[2,3],N=[1000,3000],sigma=0.1)
	filename = "s1dat"
	with open(filename,'w+') as f:
		json.dump([X.tolist(),y.tolist()],f)
	scc_result = sparseSubspaceClustering(X,filename,numThreads=1)
	print evaluate(y,scc_result)
	ens_result = ensembleSparseSubspaceClustering(X,filename,numThreads=1)
	print evaluate(y,ens_result)
	"""
	"""
	X,y = syntheticGenerator(n=100,d=[2,3,2],N=[180,200,600],sigma=0.1)
	filename = "s2dat"
	with open(filename,'w+') as f:
		json.dump([X.tolist(),y.tolist()],f)
	"""


