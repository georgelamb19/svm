import numpy as np


def backtracking(x_k, p_k, f, df, alpha0, opts_c1, rho):
    """
    INPUTS
        x_k: intial point
        p_k: search direction
        f, fd1: function and grad handlers
        alpha0: initial step length
        c1: constant in sufficient decrease condition
        rho: backtracking step length reduction factor
    RETURNS:
        alpha: final step length
        info: step length history
        
    Based on algorithm 3.1, Nocedal & Wright
    """
    
    # initialise alpha and alphas (step lengths history)
    alpha = alpha0
    info = [alpha0]
    
    # backtracking loop
    while f(x_k+alpha*p_k) > f(x_k) + opts_c1*alpha*  p_k.T@df(x_k)[:x_k.shape[0],:]:
        alpha = rho*alpha
        info += [alpha]        

    return alpha


def newtonLS(x0, c_f, c_df, c_d2f, tolNewton, maxIterNewton):
    """
    newtonLS (inner iterations of barrier method)

    Parameters
    ----------
    x0 : (d,1) initial point
    c_f : objective function
    c_df : grad fucntion
    c_d2f : hess function
    tolNewton : tolerance
    maxIterNewton : max iterations

    Returns
    -------
    x_k : (d,1) minimum
    c_f(x_k): objective evalutated at min
    nIter : number of iterations taken
    
    Based on algorithm 10.1, Boyd and Vandenberghe
    """

    nIter = 0
    x_k  = x0
    d    = x0.shape[0]
    lambda_k = 100
    
    # ls params
    alpha0 = 1
    opts_c1 = 1e-4
    rho = 0.5
    
    while (nIter < maxIterNewton) and (0.5*lambda_k**(2) > tolNewton):

        nIter += 1
        
        # compute descent direction
        try:
            p_k = -np.linalg.solve(c_d2f(x_k), c_df(x_k))
        except:
            p_k = -np.linalg.pinv(c_d2f(x_k))@c_df(x_k)
        if p_k.T@c_df(x_k)>0:
            # force descent direction if F.d2f(x_k) not positive definite
            p_k = -p_k       
        delta_x_k = p_k[:d,:]
                
        # decrement
        lambda_k = np.squeeze(delta_x_k.T@c_d2f(x_k)[:d,:d]@delta_x_k)**0.5
            
        alpha_k = backtracking(x_k, delta_x_k, c_f, c_df, alpha0, opts_c1, rho)    
        x_k += alpha_k*delta_x_k               

            
    return x_k, c_f(x_k), nIter







