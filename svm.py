# import libraries
import numpy as np
from numpy import exp
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

# import custom modules
from barrier import barrier_method
from smo import smo



# helper functions
def poly_kernel(x1, x2, d):
    "polynormial kernel function"
    return (np.matmul(x1, x2.T)+1) ** d

def rbf_kernel(p, q, gamma):
    "RBF kernel function"
    return exp(-gamma*(-2*np.matmul(p,q.T) + np.sum(p*p,1)[:,None] + np.sum(q*q,1)[None,:]))

def feasible_start(y_labels, C):
    "finds strictly feasible starting point for log barrier method"
    total_num = len(y_labels)
    pos = np.sum(y_labels==1)
    pos_frac = pos / (total_num - pos)
    a0 = np.zeros(total_num)
    for i in range(total_num):
        if y_labels[i] == 1:
            a0[i] = C * (1 - pos/total_num)
        else:
            a0[i] = C * pos_frac * (1 - pos/total_num)
    return a0



# SVM class
class SVM():
    """
    SVM class running barrier, SMO or CVXOPT optimisation algos
    Class params listed under __init__ function
    """

    def __init__(self, C, kernel, kernel_param, optimiser, mu=20, seed=True):

        self.C = C
        self.kernel = kernel
        self.kernel_param = kernel_param
        self.optimiser = optimiser
        self.mu = mu
        self.seed=seed

    def fit(self, X, y):
        """
        Function to fit SVM
        Inputs: 
        - (m,n) array of training data X
        - (m) length vector of labels y (+1 of -1)
        Each method computes:
        - sv: 1D logical vector of sv positions relative to length of X
        - a: 1D vector of sv alpha values
        - b: scalar offset
        """
        
        # store training data for predict function
        self.X_t = X
        self.y_t = y
        
        m, n = X.shape # X dimensions
        y = y.reshape(-1,1) * 1.0 # reshape y and convert to float
        
        # compute kernel matrix
        if self.kernel == 'rbf':
            K = rbf_kernel(X, X, self.kernel_param)
        elif self.kernel == 'poly':
            K = poly_kernel(X, X, self.kernel_param)



        if self.optimiser == 'barrier':
        ###################
        ##### BARRIER #####
        ###################
            
            # inputs to of quadratic optimisation problem
            P = K*y.T*y # m x m kernel matrix
            q = -np.ones((m,1)) # m vector
            G = np.vstack((np.eye(m)*-1,np.eye(m))) # 2m x m matrix
            h = np.vstack((np.zeros((m,1)), np.ones((m,1)) * self.C)) # 2m length vector
            A = y.T # 1 x m matrix
            b = np.zeros((1,1))
            
            # parameters
            x_0 = feasible_start(np.squeeze(y), self.C).reshape(-1,1)
            t_0 = 1
            mu = self.mu
            tol = 1e-6
            maxIter = 100
            
            # optimise to find alphas
            alphas, info = barrier_method(P, q, G, h, A, b, x_0, t_0, mu, tol, maxIter)



        # elements of quadratic optimisation problem ###
        P = K*y.T*y # m x m kernel matrix
        q = -np.ones(m) # m vector
        G = np.vstack((np.eye(m)*-1,np.eye(m))) # 2m x m matrix
        h = np.hstack((np.zeros(m), np.ones(m) * self.C)) # 2m length vector
        A = y.T # 1 x m matrix
        b = np.zeros(1) # single element [0]
        
        
        
        if self.optimiser == 'smo':
        ###################
        ####### SMO #######
        ###################
            
            # parameters
            tol = 1e-3
            max_passes = 5
            
            # optimise to find alphas
            alphas, info = smo(self.C, K, np.squeeze(y), tol, max_passes, self.seed)

                
        
        if self.optimiser == 'cvxopt':
        ###################
        ###### CVXOPT #####
        ###################
        
            # prepare inputs for cvxopt
            P = cvxopt_matrix(P)
            q = cvxopt_matrix(-np.ones((m, 1)))
            G = cvxopt_matrix(G)
            h = cvxopt_matrix(h)
            A = cvxopt_matrix(A)
            b = cvxopt_matrix(b)
        
            # optimise to find alphas
            cvxopt_solvers.options['show_progress'] = False
            sol = cvxopt_solvers.qp(P, q, G, h, A, b);
            alphas = np.array(sol['x'])
            info = 0
        
        
        
        ##### compute sv, a, b #####
        y = np.squeeze(y)
        alphas = np.squeeze(alphas)
        self.sv = np.squeeze(alphas > 1e-5) # support vectors
        self.a = np.squeeze(alphas[self.sv]) # sv alphas
        self.b = (y[self.sv] - np.sum(alphas*y*K,1)[self.sv])[0] # b, offset
        
        return info
    
    
    def predict(self, X):
        """
        Function to predict labels based on SVM
        Inputs:
        - X: (m,n) data for which we wish to predict labels
        Returns 1D vector of predicted labels for X
        """
        
        if self.kernel == 'rbf':
            K = rbf_kernel(self.X_t[self.sv], X, self.kernel_param)
        elif self.kernel == 'poly':
            K = poly_kernel(self.X_t[self.sv], X, self.kernel_param)
        score = np.sum(self.y_t[self.sv].reshape(-1,1)*self.a.reshape(-1,1)*K, 0) + self.b
        y_pred = np.where(score <= 0, -1, 1)
    
        return y_pred
    
    
    
    
    