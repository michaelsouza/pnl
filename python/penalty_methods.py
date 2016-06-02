import numpy as np
from numpy.linalg import norm
from numpy import inner
from numpy import max
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab

def linesearch(f,x,d,g=None):
    # linesearch (using Armijo)    
    coeff_reduction = 0.9
    coeff_step      = 0.9

    fx,dfx = f(x)
    fa     = fx
    a = 1.0
    m = coeff_reduction * inner(d, dfx)
    if(m > 0):
        raise RuntimeError('The direction is not viable.')
    done   = False
    viable = True
    if(g is not None):
        ga,_   = g(x)
        viable = np.all(ga > 0)
    
    if(not viable):
        raise RuntimeError('The start point is not viable.')

    while(not done):
        xnew = x + a * d
        if(g is not None):
            ga,_   = g(xnew)
            viable = np.all(ga > 0)
        if(viable):
            fa,dfa = f(xnew)
        done = fa <= m * a + fx
        a    = coeff_step * a
    return(xnew,fa,dfa)

def penalty_fobj(f,x,g,mu):
    gx,dgx = g(x)
    fx,dfx = f(x)

    px  = fx  - inner(np.log(gx), mu)
    dpx = dfx - np.dot(dgx, mu / gx)

    return (px, dpx) 


def penalty_logarithmic(f,x,g,verbose=False):
    # min f(x)
    # s.a g(x) >= 0,
    # where
    #  f is the objective function (f:Rn->R)
    #  g is the constraint function, i.e, x is viable if g(x) > 0 (g:Rn->Rm)

    n = len(x)
    m = len(g(x))
    k = np.ones(n,dtype=float)

    # check viability of x
    viable = np.all(g(x) > 0)
    
    if(not viable):
        raise ValueError('The start point is not viable.')
    elif(verbose):
        print('The start point is viable.')    
    
    coeff_reduction = 0.9
    mu = 1E+3
    maxit = 1000;
    X = np.zeros((n,maxit), dtype=float)
    for it in range(maxit):
        fobj = lambda x: penalty_fobj(f,x,g,mu)
        mu   = coeff_reduction * mu

        X[:,it] = x

        fx,_ = f(x)
        gx,_ = g(x)

        if(verbose):
            print('iteration %d f(x) = %lf, g(x) = %lf'%(it,fx,gx))

        # get initial values
        (fx,dx) = fobj(x)

        
        # get descend direction
        d = -dx
        xold = x
        while(True):
            (xnew,fxnew,dxnew) = linesearch(fobj,x,d,g)
            
            # stop criteria
            if (norm(xnew - x)/max([1,norm(x)]) < 0.0001): break
            
            # update x and direction
            x = xnew
            d = -dxnew
        if (norm(xold - x)/max([1,norm(xold)]) < 0.0001): break
           
    X = X[:,:it];
    fx,_ = f(x)
    gx,_ = g(x) 
    print('x    = (%f, %f)' % (x[0],x[1]))
    print('f(x) = %f' % (fx))
    print('g(x) = %f' % (gx))
    
    return (xnew,fxnew,X)
    
def fobj(x):
    fx = (x[0]-2)**2 + (x[1]-2)**2
    dx = np.array([2*(x[0]-2),2*(x[1]-2)])
    
    return (fx, dx)
    
def fcon(x):
    gx = -(x[0]**2 + 2 * x[1]**2 - 1)
    dx = -np.array([2*x[0], 4*x[1]])
    return gx,dx

x = np.array([0,0])
(x,fx,X) = penalty_logarithmic(fobj,x,fcon,verbose=True)

plt.plot(X[0],X[1],'o-')

h = 0.025
x = np.arange(-5,5,h)
y = np.arange(-5,5,h)
X, Y = np.meshgrid(x,y)
m,n  = X.shape
Z = np.zeros((m,n), dtype=float)
for i in range(m):
    for j in range(n):
        Z[i,j],_ = fobj( (X[i,j],Y[i,j]) )

CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)

for i in range(m):
    for j in range(n):
        Z[i,j],_ = fcon( (X[i,j],Y[i,j]) )
CS = plt.contour(X, Y, Z)
manual_locations = [(1,0)]
plt.clabel(CS, inline=1, fontsize=10, manual=manual_locations)
plt.show()