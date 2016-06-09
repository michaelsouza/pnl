## module goldSearch
import numpy as np

def newton_raphson(f, x,tol=1.0e-9,gradobj=False,verbose=False):
  if verbose:
    print 'Init -------'
    print 'x    = ', x
    print 'F(x) = ', f(x)

  def jacobian(f,x):
    h = 1E-04
    n = x.size
    jac = np.zeros((n,n), dtype=np.float)
    f0 = f(x)
    for i in range(n):
      temp = x[i]
      x[i] = temp + h
      f1 = f(x)
      x[i] = temp
      jac[:,i] = (f1 - f0)/h
    return jac, f0

  maxit = 30

  for i in range(maxit):
    jac, f0 = jacobian(f, x)
    if np.linalg.norm(f0)/x.size < tol: break
    dx = np.linalg.solve(jac, -f0)
    x = x + dx    
    if np.linalg.norm(dx) < tol * max(max(np.abs(x)),1): break      

  if i < maxit:
    print 'Final -----'
    print 'x     = ', x
    print 'F(x)  = ', f(x)
    print 'niter = ', i + 1
    return x
  
  raise Exception('Too many iterations')

def bracket(f,x1,h):
     ''' a,b = bracket(f,xstart,h)
         Finds the brackets (a,b) of minimum point of the user-supplied scalar 
         function f(x). The search downhill from xstart with a 
         step length h.
     '''  
     c = 1.618033989
     f1 = f(x1)
     x2 = x1 + h; f2 = f(x2)
     # get direction
     if f2 > f1:
          h  = -h
          x2 = x1 + h; f2 = f(x2)
          # check if minimum between x1-h and x1 + h
          if f2 > f1: return x2, x1-h
     # search loop
     for i in range(100):
          h = c * h
          x3 = x2 + h; f3 = f(x3)
          if f3 > f2: return x1,x3
          x1 = x2; x2 = x3
          f1 = f2; f2 = f3
     raise Exception('Bracket did not find a minimum')

def golden(f,a,b,tol=1E-03):
     ''' x,fmin = golden(f,a,b,tol=1E-06)
         Golden search method for determining x that minimizes the user-suplied
         scalar function f(x).
         The minimum must be bracketed in (a,b).
     '''
     niter = int(-2.078087 * np.log(tol/abs(b-a)))
     R = 0.618033989
     C = 1.0 - R
     # first telescoping
     x1 = R*a + C*b; x2 = C*a + R*b
     f1 = f(x1); f2 = f(x2)
     # main loop
     for i in range(niter):
          if(f1 > f2):
               a = x1
               x1 = x2; f1 = f2
               x2 = C*a + R*b; f2 = f(x2)
          else:
               b = x2
               x2 = x1; f2 = f1
               x1 = R*a + C*b; f1 = f(x1)
     if(f1 < f2): return x1,f1,i
     else: return x2,f2,i
     
def powell(F,x,h=0.1,tol=1E-06,search='golden',verbose=False):
     def f(s): return F(x + s * v)   # F in direction of v     
     n  = len(x)                     # number of variables
     df = np.zeros(n,dtype=float)    # decreases of F
     u  = np.eye(n,dtype=float) # vectors v are stored by row

     if verbose:
          print 'Init -----------'
          print 'F(x)  = ', F(x)
          print 'x     = ', x

     # set search algorithm
     if search is 'golden':
          search = golden     
     else:
          raise Exception('Unsupported linear search algorithm %s'%(search))

     print 'Steps ----------'
     for j in range(30):
          xold = x.copy()
          fold = F(xold)
          if verbose: print 'iter %5d F(x) = % g'%(j, fold)
          
          # first n line searches record decreases of F
          for i in range(n):
               v = u[i]
               a, b = bracket(f,0.0,h)
               s, fmin, niter = search(f, a, b)
               df[i] = fold - fmin
               fold = fmin
               x = x + s * v
               if verbose: print '  search direcion %2d niters = %2d df = % g'%(i,niter,df[i])
          
          # last line search in the cycle
          v = x - xold
          a,b = bracket(f,0.0,h)
          s, flast, _ = search(f,a,b)
          x = x + s * v

          # check convergence
          if np.linalg.norm(x - xold)/n < tol:
              msg = 'powell converged after %d iterations'%(j+1)
              if verbose:
                   print 'Final -----------'
                   print 'F(x)  = ', fold
                   print 'x     = ', x
                   print 'niter = ', j+1
                   print 'msg   = ', msg
              return x, j+1, msg

          # identify biggest decrease & update v's
          imax = int(np.argmax(df))
          for i in range(imax, n-1):
               u[i] = u[i+1]
               u[n-1] = v
     raise Exception('Powell method did not converge')


def __test_powell():
  F = lambda x: 100 * (x[0] - x[1]**2)**2 + (1 - x[0])**2
  xstart = np.array([3, 9])
  xmin, niter, msg = powell(F, xstart, verbose=True)  

def __test_newton_raphson():
  def F(x):
    f = np.zeros(3, dtype=np.float)
    f[0] = np.sin(x[0]) + x[1]**2 + np.log(x[2]) - 7.0
    f[1] = 3.0*x[0] + 2.0*x[1] - x[2]**3 + 1.0
    f[2] = x[0] + x[1] + x[2] - 5.0
    return f

  x = np.array([1.0, 1.0, 1.0])
  print newton_raphson(F, x, verbose=True)


if __name__ == '__main__':
  # __test_powell()
  __test_newton_raphson()