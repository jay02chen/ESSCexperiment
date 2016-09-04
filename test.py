import numpy as np
import scipy
import json
from l1regls import l1regls
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from cvxopt import matrix
import networkx as nx
import sys
import io
"""
with open("SR_sub",'r') as f:
	C = json.load(f)
"""

def parseCMUMotionData(filename):
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

def solveL1NormWithRelaxation(A,b):
	return l1regls(A,b)

def solveL1NormExactly(A,b):
	""" 
	minimize  ||x||_1
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

def constructSR(X,zeroThreshold=1e-10,aprxInf=3e+2):
	C = []
	for n in xrange(len(X)):
		A = X
		A = np.delete(A,n,axis=0)
		lambd = aprxInf 							  # Approximate to the l1-norm minimization with equality constraint
		w = solveL1NormWithRelaxation(matrix(A).T*lambd**2,matrix(X[n])*lambd**2)
		lambd = 0.25/np.sum(np.abs(np.array(w)))  # Estimate the dimension of subspace and find a proper lambda
		w = solveL1NormWithRelaxation(matrix(A).T*lambd**2,matrix(X[n])*lambd**2)
		c = [max(np.sign(abs(s)-zeroThreshold)*s,0) for s in w]
		print n,lambd
		c.insert(n,0)
		C.append(c)
	return C
def constructAffinityGraph(C):
	G = nx.Graph()
	for i in xrange(len(C)):
		for j in xrange(i,len(C)):
			G.add_edge(i,j,weight=abs(C[i][j])+abs(C[j][i]))
	return G

def spectralClustering(G):
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
	return result

def sparseSubspaceClustering(X,filename,zeroThreshold=1e-10,aprxInf=3e+2):
	C = constructSR(X,zeroThreshold,aprxInf)
	with open("SR_"+filename,'w+') as f:
		json.dump(C,f)
	result = spectralClustering(constructAffinityGraph(C))
	with open("result"+filename,'w+') as f:
		json.dump(result,f)

def subSampling(S,T=set()):
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

if __name__ == "__main__":
	filename = sys.argv[1]
	X = parseCMUMotionData(filename)
	X = np.array(X)
	X = normalize(X,axis=1)
	sparseSubspaceClustering(X,filename,zeroThreshold=1e-10,aprxInf=3e+2)
