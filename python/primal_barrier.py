import numpy as np
from numpy.linalg import solve
from numpy.linalg import norm
from numpy import dot
from numpy import diag
from scipy.optimize import minimize_scalar

def primal_barrier(A, b, c, x):
	print 'c = ', c
	print 'A = ', A
	print 'b = ', b
	print 'x = ', x

	mu = 1
	mu_tol = 1E-8
	At = A.T
	while mu > mu_tol:
		# find Newton direction
		X2 = diag(x**2)
		M = dot(dot(A, X2), At)
		u = mu * b - dot(A, dot(X2, c))
		l = solve(M, u)
		p = x - dot(X2, (c + dot(At, l)) / mu)

		# line search
		f = lambda a : dot(c, x + a * p) - mu * np.sum(np.log(x + a * p))
		d = lambda a : dot(c - mu / (x + a * p), p)
		h = lambda a : mu * np.sum( (p/(x + a * p))**2 )
		
		opt = minimize_scalar(f, method='bounded', bounds=(0,norm(p)))
		a = opt.x

		# updates
		mu *= 0.9
		x = x + a * p

		# check viability
		if any(x < 0): raise Exception('The current point is not viable')
		print 'x = ', x
		print 'f = ', np.sum(dot(c, x))
		print 'norm(A * x - b) = ', norm(A.dot(x) - b)

def __test_primal_barrier__():
	A = np.array([[1,2,3], [1,1,1]])
	x = np.ones(3)
	b = A.dot(x)
	c = np.array([1,2,1])
	primal_barrier(A,b,c,x)


if __name__ == '__main__':
	__test_primal_barrier__()