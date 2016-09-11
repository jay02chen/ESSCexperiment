1import essc
from essc import *
import sys
import os
import json
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import sqrt
from numpy import log10
from cvxopt import matrix
from cvxopt import solvers
from sklearn.preprocessing import normalize

def trial(args):
	dire = args[1]+"/"
	sigmaList = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
	sigmaList.extend(list(set(np.float64(range(50))/200+0.5) - set(sigmaList)))
	sigmaList = sorted(zip(range(len(sigmaList)),sigmaList),key=lambda x:x[1])
	if (len(args) > 2 and args[2] == "redo") or not os.path.exists(dire):
		print "regenerating synthetic data..."
		emptyDir(dire)
	for i, sigma in sigmaList:
		if not os.path.exists(dire+str(i)+"dat"):
			X,y,Base,meta = syntheticGenerator(n=20,d=[3,3,3,3,3,3],N=[400,100,600,300,400,1200],sigma=sigma,orthonormal=True)
			with open(dire+str(i)+"dat",'w+') as f:
				json.dump([X.tolist(),y.tolist(),Base,meta],f)
	os.chdir(dire)
	for i, sigma in sigmaList:
		subdire = "s"+str(i)+"/"
		if not os.path.exists(subdire):
			os.mkdir(subdire)
		filename = str(i)+"dat"
		argument = ["",filename,subdire,filename]
		SSCresult,ESSCresult,y = loadResult(subdire,filename)
		print ""
		print filename,":\tsigma = ",sigma
		if len(args) == 2 and len(SSCresult) > 0:
			# print "SSC",SSCresult
			# print "ESSC",ESSCresult
			# print "Ans",y
			yvSSC = evaluate(y,SSCresult)
			yvESSC = evaluate(y,ESSCresult)
			SscvEssc = evaluate(SSCresult,ESSCresult)
			print "SSC vs Ans",yvSSC
			print "ESSC vs Ans",yvESSC
			print "SSC vs ESSC",SscvEssc
			print ""
		elif os.path.exists("writing_"+filename):
			continue
		else:
			with open("writing_"+filename,'w+') as f:
				f.write("%d"%(os.getpid()))
			with open("writing_"+filename,'r') as f:
				if int(f.readline()) != os.getpid():
					continue
			subtrial(argument)
			argument = ["",filename,subdire,filename+"k","k"]
			subtrial(argument)
			os.unlink("writing_"+filename)

def subtrial(args):
	infile = args[1]
	indire = args[2]
	outfile = args[3]
	SRinfile = indire+"SR_"+infile
	with open(infile,'r') as f:
		X = json.load(f)
		y = X[1]
		sigma = X[3][3]
		X = X[0]
	if len(args) > 4 and (args[4] == "k" or args[4] == "K"):
		K = len(set(y))
	else: K = -1
	print "K = ",K
	X = np.array(X)
	X = normalize(X,axis=1)
	subSamples = subSampling(range(len(X)))
	A = dict()
	Weight = dict()
	for i in xrange(len(subSamples)):
		subsample = subSamples[i]
		if os.path.exists(indire+str(i)):
			with open(indire+str(i),'r') as f:
				C = json.load(f)
		else: 
			C = constructSR(X[subsample],sigma=sigma)
			with open(indire+str(i),'w+') as f:
				json.dump(C,f)
		subK = len(set([y[j] for j in subsample]))
		if K == -1:
			subK = -1
		result = spectralClusteringWithL(getLaplacian(C),subK)
		print i,len(subSamples),len(subsample),subK,evaluate(result,[y[s] for s in subsample])
		print {cluster:len([1 for ss in subsample if y[ss] == cluster]) for cluster in [cluster for cluster in set([y[s] for s in subsample])]}
		print ""
		C = None
		for j in xrange(len(result)):
			for k in xrange(j+1,len(result)):
				i1 = min(subsample[j],subsample[k])
				i2 = max(subsample[j],subsample[k])
				if result[j] == result[k]:
					if (i1,i2) not in A:
						A[i1,i2] = 1
					else: A[i1,i2] = A[i1,i2] + 1
				if (i1,i2) not in Weight:
					Weight[i1,i2] = 1
				else: Weight[i1,i2] = Weight[i1,i2] + 1
	G = nx.Graph()
	for i in xrange(len(X)):
		G.add_node(i)
	for edge in A:
		if A[edge]*2 >= Weight[edge]:  		#majority voting
			G.add_edge(edge[0],edge[1])
	A = None
	ESSCresult = spectralClustering(G,K)
	if os.path.exists(SRinfile):
		with open(SRinfile,'r') as f:
			C = json.load(f)
	else:
		C = constructSR(X,sigma=sigma)
		with open(SRinfile,'w+') as f:
			json.dump(C,f)
	SSCresult = spectralClusteringWithL(getLaplacian(C),K)
	with open(indire+"SSCres_"+outfile,'w+') as f:
		json.dump(SSCresult,f)
	with open(indire+"ESSCres_"+outfile,'w+') as f:
		json.dump(ESSCresult,f)
	print "SSC",SSCresult
	print "ESSC",ESSCresult
	print "Ans",y
	print "SSC vs Ans",evaluate(y,SSCresult)
	print "ESSC vs Ans",evaluate(y,ESSCresult)
	print "SSC vs ESSC",evaluate(SSCresult,ESSCresult)

def trial_FDdistribution(args):
	outdire = args[1]
	if outdire[-1] != "/":
		outdire = outdire + "/"
	emptyDir(outdire)
	os.chdir(outdire)
	Sigma = [0.3,0.8]
	Trials = 300
	Dimension = np.array([2,2])
	Points = np.array([10,10])
	Distr = []
	Weight = []
	Pos = []
	for sigma in Sigma:
		for exp in xrange(4):
			distr = []
			enum = 0
			for t in xrange(Trials):
				X,y,Base,meta = syntheticGenerator(n=10,d=Dimension,N=Points,sigma=sigma,orthonormal=True)
				xi = X[0]
				X_mi = X[1:,:]
				lambd = np.float64(1)/sqrt(Dimension[y[0]])
				lambd = sqrt(0.5*lambd)
				c = l1regls(matrix(X_mi).T*lambd,matrix(xi)*lambd)
				c = np.array(c).reshape(-1)
				c = log10(abs(c))
				# c = abs(c)
				for i in xrange(len(c)):
					key = c[i]
					if y[i+1] == y[0]:
						distr.append(key)
					else: enum = enum + 1
			pos = np.float64(len(distr))/(len(distr)+enum)
			dw = np.ones_like(distr)*pos/len(distr)
			Distr.append(distr)
			Weight.append(dw)
			Pos.append(pos)
			Points = Points * 10
		maximum = int(np.ceil(max([max(d) for d in Distr])))
		minimum = int(np.floor(min([min(d) for d in Distr])))
		rg = np.linspace(minimum,maximum,24)
		plt.hist(Distr[0],bins=rg,weights=Weight[0], facecolor='blue',label="10 pts",alpha=1)
		plt.hist(Distr[1],bins=rg,weights=Weight[1], facecolor='green',label="100 pts",alpha=0.8)
		plt.hist(Distr[2],bins=rg,weights=Weight[2], facecolor='red',label="1000 pts",alpha=0.7)
		plt.hist(Distr[3],bins=rg,weights=Weight[3], facecolor='burlywood',label="10000 pts",alpha=0.7)
		plt.legend()
		plt.grid()
		plt.xlabel('log(abs(c))')
		plt.title("sigma = "+str(sigma))
		plt.savefig("sigma"+str(sigma*10)+".png")
		with open("sigma"+str(sigma*10)+"_pos",'w+') as f:
			json.dump(Pos,f)
	plt.show()



if __name__ == "__main__":
	args = [s for s in sys.argv]
	# trial(args)
	trial_FDdistribution(args)
