import numpy as np
from numpy.linalg import norm
from numpy import inner
from numpy import max

def linesearch(f,x,d):
    # linesearch (using Armijo)    
    coeff_reduction = 0.9
    coeff_step      = 0.9

    fx,gx = f(x)
    a = 1.0
    m = coeff_reduction * inner(d, gx)
    if(m > 0):
        raise RuntimeError('The direction is not viable.')
    done = False
    while(not done):
        xnew  = x + a * d
        fa,ga = f(xnew)
        done  = fa <= m * a + fx
        a     = coeff_step * a
    return(xnew,fa,ga)

def penalty_logarithmic(f,x,g,verbose=False):
    # min f(x)
    # s.a g(x) <= 0,
    # where
    # f is the objective function (f:Rn->R)
    # g is the constraint function, i.e, x is viable if g(x) < 0 (g:Rn->Rm)

    n = len(x)
    m = len(g(x))
    k = np.ones(n,dtype=float)

    # check viability of x
    viable = not np.any(g(x) > 0)
        
    if(not viable):
        raise ValueError('The start point is not viable.')
    elif(verbose):
        print('The start point is viable.')    
            
    # get initial values
    (fx,dx) = f(x)
    
    # get descend direction
    d = -dx

    done = False
    while(not done):
        print(x)
        (xnew,fxnew,dxnew) = linesearch(f,x,d)
        
        # stop criteria
        if (norm(xnew - x)/max([1,norm(x)]) < 0.0001): break
        
        # update x and direction
        x = xnew
        d = -dxnew
    
    return (xnew,fxnew)
    

def fobj(x):
    fx = (x[0] + 1) **2 + x[1]**2
    dx = np.array([2*(x[0] + 1),2*x[1]])
    
    return (fx, dx)
    
def fcon(x):
    gx = np.array([-x[0], -x[1]])
    return gx
  
x = np.array([5,5])
penalty_logarithmic(fobj,x,fcon,verbose=True)

    
    
    

    

    