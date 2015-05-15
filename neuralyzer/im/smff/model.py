'''
An implementation of Calcium Signal extraction, demixing and ROI estimation
along the model and algorithm described in 

> A structured matrix factorization framework for large scale calcium imaging
> data analysis.

by Pnevmatikakis et al. 2014

'''

import numpy as np

from neuralyzer.im import nmf

#from neuralyzer.utils import log




# MODEL INITIALIZATION CODE
# -----------------------------------------------------------------------------

def _init_model(Y, gaussian_blur_memmap, num_components=1, ws=41):
    '''
    
    Y: the data of form [Y] = d x T
    gaussian_blur_memmap : a numpy memory map to a gaussian blur matrix

    num_components : the number of components (neurons) we aim for

    WARNING !! we assume quadratic image shape

    '''

    TINY_POSITIVE_NUMBER = np.finfo(np.float).tiny 

    d, T = Y.shape

    #mediandata = np.median(Y, axis=1)
    #R = Y - np.tile(mediandata, (Y.shape[1], 1)).T
    R = Y.copy()
    D = gaussian_blur_memmap

    ims = int(np.sqrt(d))
    # in case we would like to make it an input argument at some point ..
    ims = (ims, ims)

    A, C = [], []

    for k in range(num_components):

        rho = np.dot(D, R)
        rhomax = rho.max(axis=1)
        wcent = np.argmax(rhomax)
        wcent = (np.mod(wcent, ims[0]), int(wcent/ims[0]))
        wmask = window_mask(wcent, ims, ws).flatten()
        Rw = R[wmask, :]
        H_init = rhomax[wmask].flatten()
        W, H = _nmf_lars_greedy_init(Rw.T, H_init, iterations=30, normalize=False, alpha=0.1)
        ak = np.zeros(R.shape[0])
        ak.flat[wmask] = H
        
        A.append(ak/ak.sum()) # normalize
        C.append(W.clip(TINY_POSITIVE_NUMBER, np.inf))
        R[wmask, :] = (R[wmask, :] - np.dot(W, H.T).T).clip(TINY_POSITIVE_NUMBER, np.inf)

    # initialize background trace + signal
    # get better init estimate for background!!
    f, b = _nmf_lars_greedy_init(R.T, 'random', k=1, iterations=30, normalize=False, alpha=1.)
    f = f/f.sum()

    C = np.array(C).reshape(num_components, T) # will be of shape k x T
    A = np.array(A).T # will be of shape d x k

    return A, C, b, f 


def _nmf_lars_greedy_init(V, H_init, k=None, iterations=30, **kwargs):
    '''
    V = WH : V, W, H >= 0

    V.shape = (m, n)
    W.shape = (m, k)
    H.shape = (k, n)
    '''
    from sklearn.linear_model import LassoLars 
    m, n = V.shape
    if H_init == 'random':
        H = np.random.rand(n, k)
    else:
        H = H_init
        if H.ndim == 1:
            H.shape  += (1,) 

    # Perform 'alternating' minimization.
    for iter_num in range(iterations):
        lB = LassoLars(positive=True, max_iter=200, **kwargs)
        lB.fit(H, V.T)
        W = lB.coef_





        lA = LassoLars(positive=True, max_iter=200, **kwargs)
        lA.fit(W, V)
        H = lA.coef_

    return W, H


def window_mask(cent, ims, ws):
    ''' symmetric and odd window sizes only '''
    w = np.zeros(ims, dtype='bool')
    if ws == 0:
        return w
    if not(ws % 2):
        raise ValueError('ws has to be odd.')
    ws = int(ws/2)
    xl = np.array([cent[1]-ws, cent[1]+ws+1]).clip(0, ims[1])
    yl = np.array([cent[0]-ws, cent[0]+ws+1]).clip(0, ims[0])
    w[xl[0]:xl[1], yl[0]:yl[1]] = True
    return w



def _gauss_kernel(size, var=None):    
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

def _gaussian_blur_image(imagesize, kernelsize, kernelvariance, position):
    pos = position
    ims = imagesize
    ks = kernelsize
    kernel = _gauss_kernel(ks, kernelvariance)
    # generate a padded image
    im = np.zeros((ims[0]+ks-1, ims[1]+ks-1))
    im[pos[0]:pos[0]+ks, pos[1]:pos[1]+ks] = kernel
    # remove padding
    im = im[int(ks/2):-int(ks/2), int(ks/2):-int(ks/2)]
    return im

def _gaussian_blur_matrix(imagesize, kernelsize, kernelvariance):
    dim = imagesize[0]*imagesize[1]
    D = np.zeros((dim, dim))
    for idx in range(dim):
        pos = (int(idx/imagesize[0]), np.mod(idx, imagesize[0]))
        D[:,idx] = _gaussian_blur_image(
                imagesize, kernelsize, kernelvariance, pos
                ).flatten()
    return D



"""
DEPRECATED BUT STRUCTURALLY INTERESTING

def gaussian_blur_matrix_sparse(dim, kernelsize, kernelvariance):
    '''
    
    D is a symmetric matrix!

    tested with : plt.imshow(D.toarray()[:100,:100])
    '''
    from scipy import sparse as sps
    kernel = _gauss_kernel(kernelsize, kernelvariance).flatten()
    kernel = kernel.flatten()
    s = int(len(kernel)/2)
    data = []
    offsets = []
    for i in range(len(kernel)):
        data.append(np.ones(dim)*kernel[i])
        offsets.append(s-i)
    D = sps.dia_matrix((np.array(data), np.array(offsets)), shape=(dim, dim))
    return D
"""


# SPARSE NON-NEGATIVE MATRIX FACTORIZATION CODE
# -----------------------------------------------------------------------------

class SMFF(object):

    def __init__(self, *args, **kwargs):
        self.max_iter = kwargs.pop('max_iter', 100)
        self._step = 0
        

    def _stop(self):
        '''
        simple interation number based stop criterion for now
        '''
        return self._step > self.max_iter


    def init_model(self, ):

        C, A, b, f = _init_model(Y, **kwargs)


    def fit_model(self, Y, **kwargs):
    
        while not self._stop():
            C, f = self._update_C_f(C, A, b, f, Y)
            A, b = self._update_A_b(C, A, b, f, Y)
            
            self._step += 1


    @staticmethod
    def _update_C_f(C, A, b, f, Y):
        '''
        minimize \sum (1 G c_j)
        subject to Gc_j >= 0, ||Y(i,:) - A(i,:)C - b(i)f.T || < \sigma_i * sqrt(T)
        '''

        return C, f


    @staticmethod
    def _update_A_b(C, A, b, f, Y):
        '''
        minimize || A ||_1
        subject to : A, b >= 0, ||Y(i,:) - A(i,:)C - b(i)f.T || < \sigma_i * sqrt(T)
        '''

        return A, b 
