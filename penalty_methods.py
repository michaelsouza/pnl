import numpy as np
from numpy import inner
from numpy.linalg import norm

def armijo(f,x,d):
	# f objective function (f:Rn->R)
 	# x variables (in Rn)
 	# d descend direction (in Rn)

 	coefReduction = 0.9
 	coefAngular   = 0.9

 	fx,dx = f(x)

 	beta = coefAngular * inner(dx, d)

 	if(beta >= 0):
 		raise exception('The direction d is not descent.')

 	maxit = 1000
 	alpha = 1
 	done  = False
 	it    = 0
 	while(not done):
 		it   = it + 1
 		xnew = x + alpha * d
 		(fxnew,dxnew) = f(xnew)
 		done = fxnew < beta * alpha + fx
 		alpha = coefReduction * alpha
 		if(it > maxit):
 			raise exception('The max number of iteration was exceeded.')

 	return (xnew,fxnew,dxnew)

def gradient(f,x):
 	(fx,dx) = f(x)
 	d    = -dx
 	done = False
 	xtol = 0.0001
 	while(not done):
 		(xnew, fxnew, dxnew) = armijo(f,x,d)
 		
 		# stop criteria
 		done = norm(x - xnew) / max([1, norm(x)]) < xtol

 		# update 
 		x = xnew
 		d = -dxnew

 	return (xnew,fxnew)

def fobj(x):
	fx = (x[0] - 1)**2 + x[1]**2
	dx = np.array([2*(x[0] - 1), 2*x[1]])
	return (fx,dx)


x = np.array([5,5])

(x,fx) = gradient(fobj,x)
print('x = [%f,%f]'%(x[0],x[1]))
print('f = %f'%(fx))


