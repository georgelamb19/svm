import numpy as np
import copy
from collections import defaultdict
from newtonLS import newtonLS



def f(x, P, q):
    "original objective"
    return np.squeeze(0.5*x.T@P@x + q.T@x)

def df(x, P, q):
    "grad of original objective"
    return P@x + q

def d2f(x, P, q):
    "hess of original objective"
    return P
    
def phi(x, G, h):
    "log barrier"
    return -np.sum(np.log(h - G@x))

def dphi(x, G, h):
    "grad of log barrier"
    tmp = 1 / (h - G@x)
    return G.T@tmp

def d2phi(x, G, h):
    "hess of log barrier"
    tmp = np.squeeze(1 / (h - G@x))
    return G.T@(np.diag(tmp)**2)@G   



def barrier_method(P, q, G, h, A, b, x0, t, mu, tol, max_iter):
    """
    Barrier method (outer iterations)
    
    Inputs:
    -----------
    P: (d,d) 
    q: (d,1)    
    G: (m,d)
    h: (m,1)            
    A: (p,d)
    b: (p,1)
    x0: (d,1)
    
    Returns:
    -----------
    optimal x_k (d,1)
    info (iterate and duality gap history)
    
    Based on algorithm 11.1, Boyd and Vandenberghe
    """

    m = G.shape[0]
    p = A.shape[0]
    
    # initialise
    nIter = 0
    x_k = x0
    
    # initialise info
    info = defaultdict(list)
    info['iterates'].append(copy.deepcopy(x_k))
    
    tolNewton = 1e-12
    maxIterNewton = 100

    while (m/t >= tol) and (nIter < max_iter):
        
        duality_gap = m/t
        
        c_f = lambda x: t*f(x, P, q) + phi(x, G, h)
        c_df = lambda x: np.vstack([t*df(x, P, q) + dphi(x, G, h), np.zeros((p,1))])
        c_d2f = lambda x: np.vstack([np.hstack([t*d2f(x, P, q) + d2phi(x, G, h), A.T]), np.hstack([A, np.zeros((p,p))])])  
        
        x_k, fLS, nIterLS = newtonLS(x_k, c_f, c_df, c_d2f, tolNewton, maxIterNewton)

        # increase t
        t *= mu

        # increment number of iterations
        nIter += 1
        
        # update info
        info['iterates'].append(copy.deepcopy(x_k))
        info['duality_gaps'] += int(nIterLS) * [copy.deepcopy(duality_gap)]

    return x_k, info





