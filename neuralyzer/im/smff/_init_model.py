'''
Model Initialization Code.

Since NMF has not a unique solution, the outcome of the optimization procedure
depends on the initialization procedure. Random initialization is typically not
optimal.

Here we implemented a fast heuristic model initialization procedure along the
original reference by Pnevmatikakis et al. 2014.
'''

import numpy as np 

from .. import nmf 

from neuralyzer import log
logger = log.get_logger()

TINY_POSITIVE_NUMBER = np.finfo(np.float).tiny 


def _init_model(Y, gaussian_blur_memmap, num_components=1, ws=41,
        spl0_comps=0.8, spl0_bg=0.4, iterations=10):
    '''
    Rather heuristic initialization procedure.
    
    Y: the data of form [Y] = d x T
    gaussian_blur_memmap : a numpy memory map to a gaussian blur matrix

    num_components : the number of components (neurons) we aim for

    WARNING !! we assume quadratic image shape

    '''
    logger.info('Initializing SMFF model: %s components' % num_components)

    d, T = Y.shape

    #mediandata = np.median(Y, axis=1)
    #R = Y - np.tile(mediandata, (Y.shape[1], 1)).T
    R = Y.copy()
    f, b = _nmf_l0(R.T, spl0=spl0_bg)

    # subtract background ..
    #b_norm = b/b.sum()
    #R = R - np.dot(np.dot(R.T, b_norm.T), b_norm).clip(TINY_POSITIVE_NUMBER, np.inf).T
    R  = (R - np.dot(f, b).T).clip(TINY_POSITIVE_NUMBER, np.inf)

    D = gaussian_blur_memmap

    ims = int(np.sqrt(d))
    # in case we would like to make it an input argument at some point ..
    ims = (ims, ims)

    A, C = [], []

    for k in range(num_components):

        logger.info('component %s / %s ' % (k+1, num_components))

        rho = np.dot(D, R)
        rhomax = rho.max(axis=1)
        wcent = np.argmax(rhomax)
        wcent = (np.mod(wcent, ims[0]), int(wcent/ims[0]))
        wmask = window_mask(wcent, ims, ws).flatten()
        Rw = R[wmask, :]

        H_init = rhomax[wmask].flatten()
        H_init /= H_init.sum() # normalize
        H_init.shape += (1,)
        W_init = np.dot(Rw.T, H_init)

        W, H = _nmf_l0(Rw.T, W_init, spl0=spl0_comps, iterations=iterations)

        ak = np.zeros(R.shape[0])
        ak.flat[wmask] = H
        ak = (ak/ak.sum()).clip(TINY_POSITIVE_NUMBER, np.inf) # normalize
        A.append(ak) # normalize

        c = np.dot(R.T, ak)
        #W = W.clip(TINY_POSITIVE_NUMBER, np.inf)
        C.append(c/c.sum())

        R[wmask, :] = (R[wmask, :] - np.dot(W, H).T).clip(TINY_POSITIVE_NUMBER, np.inf)

    # initialize background trace + signal
    # get better init estimate for background!!
    #f, b = _nmf_lars_greedy_init(R.T, 'random', k=1, iterations=30, normalize=False, alpha=1.)
    #f, b = _nmf_l0(R.T, alpha=0.01)

    R  = (R + np.dot(f, b).T).clip(TINY_POSITIVE_NUMBER, np.inf)

    f, b = _nmf_l0(R.T, spl0=spl0_bg, iterations=iterations)

    #f = f/f.sum()

    C = np.array(C).reshape(num_components, T) # will be of shape k x T
    A = np.array(A).T # will be of shape d x k

    return A, C, b.T, f.T


def _nmf_l0(V, W_init=None, spl0=0.6, iterations=10, k=1):
    '''
    '''
    nmfl0 = nmf.NMF_L0(spl0=spl0, iterations=iterations)
    nmfl0.fit(V, W_init=W_init, k=k)
    return nmfl0._W, nmfl0._H


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
