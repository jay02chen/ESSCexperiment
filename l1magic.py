import numpy as np
from numpy import ceil
from scipy import log
from scipy import sqrt
from scipy.linalg import norm

def l1qc_newton(x0,u0,A,b,epsilon,tau,newtontol=np.float64(1e-3),newtonmaxiter=50,cgtol=np.float64(1e-8),cgmaxiter=200):
	# line search params
	f64one = np.float64(1.)
	alpha = np.float64(0.01)
	beta = np.float64(0.5)
	AtA = np.float64(A.T.dot(A))
	# initial point
	x = np.float64(x0)
	u = np.float64(u0)
	r = np.float64(A.dot(x) - b)
	fu1 = np.float64(x-u)
	fu2 = np.float64(-x-u)
	fe = np.float64(0.5)*(r.T.dot(r) - epsilon**2)
	f = np.float64(sum(u) - (f64one/tau)*(sum(log(-fu1)) + sum(log(-fu2)) + log(-fe)))
	niter = 0
	done = False
	while not done:
		atr = A.T.dot(r)
		ntgz = f64one/fu1 - f64one/fu2 + f64one/fe*atr
		ntgu = -tau - f64one/fu1 - f64one/fu2
		gradf = -(f64one/tau)*np.hstack((ntgz,ntgu))

		sig11 = f64one/fu1**2 + f64one/fu2**2
		sig12 = -f64one/fu1**2 + f64one/fu2**2
		sigx = sig11 - np.float64(sig12**2)/sig11

		w1p = ntgz - np.float64(sig12)/sig11*ntgu
		H11p = np.diag(sigx) - (f64one/fe)*AtA + (f64one/fe)**2*atr.dot(atr.T)
		dx = np.linalg.solve(H11p,w1p)
		Adx = A.dot(dx)
		du = (f64one/sig11)*ntgu - (np.float64(sig12)/sig11)*dx
		# minimum step size that stays in the interior
		ifu1 = [i for i in xrange(len(dx)) if dx[i]-du[i] > 0]
		ifu2 = [i for i in xrange(len(dx)) if -dx[i]-du[i] > 0]
		aqe = Adx.T.dot(Adx)
		bqe = np.float64(2)*r.T.dot(Adx)
		cqe = r.T.dot(r) - epsilon**2
		temp = -fu1[ifu1]/(dx[ifu1]-du[ifu1])
		smax = np.float64(1)
		if len(temp) > 0:
			smax = min(smax,np.min(temp))
		temp = -fu2[ifu2]/(-dx[ifu2]-du[ifu2])
		if len(temp) > 0:
			smax = min(smax,np.min(temp))
		smax = min(smax,(-bqe+sqrt(bqe**2-4.*aqe*cqe))/(2.*aqe))
		s = np.float64(0.99)*smax

		# backtracking line search
		suffdec = False
		backiter = 0
		while not suffdec:
			xp = x + s*dx
			up = u + s*du
			rp = r + s*Adx
			fu1p = xp - up
			fu2p = -xp - up
			fep = f64one/2*(rp.T.dot(rp) - epsilon**2)
			fp = sum(up) - (f64one/tau)*(sum(log(-fu1p)) + sum(log(-fu2p)) + log(-fep))
			flin = f + alpha*s*(gradf.T.dot(np.hstack((dx,du))))
			# print fep,fp,flin
			suffdec = (fp <= flin)
			s = beta*s 
			backiter = backiter + 1
			if backiter > 48:
				# print "Stuck on backtracking line search, returning previous iterate."
				xp = x 
				up = u 
				break
		# set upt for next iteration
		x   = xp
		u   = up 
		r   = rp 
		fu1 = fu1p
		fu2 = fu2p 
		fe  = fep
		f   = fp 
		lambda2 = np.float64(-(gradf.T.dot(np.hstack((dx,du)))))
		stepsize = np.float64(s*norm(np.hstack((dx,du))))
		niter = niter + 1
		done = (lambda2/2 < newtontol) or (niter >= newtonmaxiter)
		# print "Newton iter = %d, Functional = %8.3f, Newton decrement = %8.3f, Stepsize = %8.3e"%(niter, f, lambda2/2, stepsize)
	return xp, up, niter

def l1qc_logbarrier(A,b,epsilon,lbtol=np.float64(1e-3),mu=np.float64(10),cgtol=np.float64(1e-8),cgmaxiter=200):
	f64one = np.float64(1.)
	x0 = np.float64(A.T.dot(b))
	newtontol = np.float64(lbtol)
	newtonmaxiter = 50
	N = len(x0)
	# starting point --- make sure that it is feasible
	if norm(A.dot(x0) - b) > epsilon:
		# print "Starting point infeasible; using x0 = At*inv(AAt)*y."
		w = np.float64(np.linalg.solve(A.dot(A.T),b))
		x0 = np.float64(A.T.dot(w))
	x = x0
	u = 0.95*abs(x0) + 0.10*max(abs(x0))
	# print "Original l1 norm = %.3f, original functional = %.3f"%(sum(abs(x0)), sum(u))
	# choose initial value of tau so that the duality gap after the first
	# step will be about the origial norm
	tau = max(np.float64(2*N+1)/sum(abs(x0)), 1)
	lbiter = ceil(np.float64(log(np.float64(2*N+1))-log(np.float64(lbtol))-log(tau))/log(mu))
	# print "Number of log barrier iterations = %d\n"%(lbiter)
	totaliter = 0
	for ii in xrange(1,int(lbiter)+1):
		xp,up,ntiter = l1qc_newton(x,u,A,b,epsilon,tau, newtontol, newtonmaxiter, cgtol, cgmaxiter)
		totaliter = totaliter + ntiter
		# print "\nLog barrier iter = %d, l1 = %.3f, functional = %8.3f, tau = %8.3e, total newton iter = %d\n"%(ii, sum(abs(xp)), sum(up), tau, totaliter)
		x = xp
		u = up
		tau = mu*tau
	return xp

"""
import numpy as np
from scipy.linalg import norm
from l1magic import l1qc_logbarrier as l1qc
A = np.random.uniform(0,1,(3,3))
b = np.random.uniform(0,1,(3))
ep = np.float64(1.)
l1qc(A,b,ep)

"""