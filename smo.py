import numpy as np
import copy
from collections import defaultdict

def smo(C, K, y, tol, max_passes, seed):
    """
    Inputs:
        C: svm parameter
        K: (m, m) kernel matrix
        y: (m) length vector of true labels
        tol: tolerance on KKT conditions
        max_passes: max number of times to iterate over alphas without a change
    
    Returns:
        alphas: (m) length vector
        b: scalar offset
        
    Based on Andrew Ng's simplified SMO algorithm (CS229 notes)
    """
    if seed:
        np.random.seed(196)
    
    m = K.shape[0]
    idx = np.arange(m)
    
    # initialise params
    alphas = np.zeros(m)
    b = 0.0
    E = np.zeros(m)
    passes = 0
    
    # initialise info
    info = defaultdict(list)
    info['iterates'].append(copy.deepcopy(alphas))
    
    
    while passes < max_passes:
        num_changed_alphas = 0
        
        for i in range(m):
            

            E[i] = np.sum(alphas * y * K[:, i]) + b - y[i]

            # if alphas[i] does not fulfill the KKT conditions to within some tol we perform update
            if ((y[i] * E[i] < -tol) & (alphas[i] < C)) | ((y[i] * E[i] > tol) & (alphas[i] > 0)):
                
                # select j!=i randomly
                j = np.random.choice(np.delete(idx,i))

                # calculate E[j] = f(x[j]) - y[j]
                E[j] = np.sum(alphas * y * K[:, j]) + b - y[j]

                # Save old alphas
                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                # Compute L and H
                if y[i] != y[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                
                # if L == H continue to next i
                if L == H:
                    continue

                # compute eta
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                
                # if eta >= 0 continue to next i
                if eta >= 0:
                    continue

                # compute and clip new value for alphas[j]
                alphas[j] = alphas[j] - (y[j] * (E[i] - E[j]) / eta)
                alphas[j] = min(H, alphas[j])
                alphas[j] = max(L, alphas[j])

                # check if change in alphas[j] is significant
                if np.abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue

                # determine value of alphas[i]
                alphas[i] = alphas[i] + y[i] * y[j] * (alpha_j_old - alphas[j])

                # compute b1 and b2
                b1 = b - E[i] - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                b2 = b - E[j] - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) * K[j, j]
                
                # Compute b
                if (0 < alphas[i]) & (alphas[i] < C):
                    b = b1
                elif (0 < alphas[j]) & (alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2
                
                num_changed_alphas += 1
                info['iterates'].append(copy.deepcopy(alphas))
                
        info['changes'].append(copy.deepcopy(num_changed_alphas))

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
            
    return alphas, info






