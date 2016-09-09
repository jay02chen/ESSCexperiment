import essc
from essc import *
import sys
import os
import json
import numpy as np
from sklearn.preprocessing import normalize

def trial(args):
	dire = args[1]
	sigmaList = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
	sigmaList = sorted(zip(range(len(sigmaList)),sigmaList),key=lambda x:x[1])
	if (len(args) > 2 and args[2] == "redo") or not os.path.exists(dire):
		print "regenerating synthetic data..."
		emptyDir(dire)
	for i, sigma in sigmaList:
		if not os.path.exists(dire+str(i)+"dat"):
			X,y,Base,meta = syntheticGenerator(n=20,d=[3,3,3,3,3,3],N=[40,10,60,30,40,120],sigma=sigma,orthonormal=True)
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
		X = X[0]
		sigma = X[3][3]
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

if __name__ == "__main__":
	args = [s for s in sys.argv]
	if sys.argv[1] == "trial1":
		args[1] = "trial1/"
		trial(args)