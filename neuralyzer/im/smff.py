'''
Implementation of the algorithm presented in

> A structured matrix factorization framework for large scale calcium imaging
> data analysis.

by Pnevmatikakis et al. 2014

'''

import numpy as np


# GREEDY INITIALIZATION CODE
# -----------------------------------------------------------------------------

def center_data(data, method='mean'):
    if method == 'mean':
        return data - np.array(data.shape[0]*[data.mean(axis=0)])
    elif method == 'median':
        return data - np.array(data.shape[0]*[np.median(data,axis=0)])
    else:
        raise ValueError('Method "%s" not available for data centering.')


def gauss_kernel(size, var=None):    
    if type(size) == int:
        size = (size, size)
    else:
        raise ValueError('We only provide symmetric kernels with sizes specified by an int.')
    if not np.mod(size[0],2) or not np.mod(size[1],2):
        raise ValueError('The size of the Kernel has to be odd.')
    s = (int(size[0]/2.), int(size[0]/2.))
    if var is None: v = s
    elif type(var) == int: v = (var, var)
    else: v = var
    x, y = np.mgrid[-s[0]:s[0]+1, -s[1]:s[1]+1]
    g = np.exp(-(x**2/(2*float(v[0]))+y**2/(2*float(v[1]))))
    return g / g.sum()


def gaussian_blur_matrix(dim, kernelsize, kernelvariance):
    kernel = gauss_kernel(kernelsize, kernelvariance).flatten()
    kl = len(kernel)
    D = np.zeros((dim, dim+kl-1))
    for idx in range(dim):
        D[idx,idx:idx+kl] = kernel
    D = D[:,((kl-1)/2):-(kl-1)/2]
    # !! D is symmetric
    return D


def gaussian_blur_matrix_sparse(dim, kernelsize, kernelvariance):
    '''
    
    D is a symmetric matrix!

    tested with : plt.imshow(D.toarray()[:100,:100])
    '''
    from scipy import sparse as sps
    kernel = gauss_kernel(kernelsize, kernelvariance).flatten()
    kernel = kernel.flatten()
    s = int(len(kernel)/2)
    data = []
    offsets = []
    for i in range(len(kernel)):
        data.append(np.ones(dim)*kernel[i])
        offsets.append(s-i)
    D = sps.dia_matrix((np.array(data), np.array(offsets)), shape=(dim, dim))
    return D
