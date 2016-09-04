import numpy as np
import json
from l1regls import l1regls
from sklearn.preprocessing import normalize
from cvxopt import matrix
import sys
import io


def readData(filename):
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

def constructSR(X,zeroThreshold=1e-8):
	C = []
	for n in xrange(len(X)):
		A = X
		np.delete(A,n,axis=0)
		l = 3e+2
		w = l1regls(matrix(A).T*l**2,matrix(X[n])*l**2)
		l = 0.25/np.sum(np.abs(np.array(w)))
		w = l1regls(matrix(A).T*l**2,matrix(X[n])*l**2)
		c = np.array(w)
		print n,l
		np.insert(c,n,0)
		C.append(c.tolist())
	return C

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
	X = readData(sys.argv[1])
	X = np.array(X)
	X = normalize(X,axis=1)
	with open("temp",'w+') as f:
		json.dump(X.tolist(),f)
	C = constructSR(X)
	with open("5_SCCSR.j",'w+') as f:
		json.dump(C,f)
