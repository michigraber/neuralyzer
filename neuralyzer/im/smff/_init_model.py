'''
Model Initialization Code.

Since NMF has not a unique solution, the outcome of the optimization procedure
depends on the initialization procedure. Random initialization is typically not
optimal.

Here we implemented a fast heuristic model initialization procedure along the
original reference by Pnevmatikakis et al. 2014.
'''

import numpy as np 
from scipy import ndimage

from joblib import Parallel, delayed

from .. import nmf 

from neuralyzer import log
logger = log.get_logger()

TINY_POSITIVE_NUMBER = np.finfo(np.float).tiny 

N_JOBS = -1


def _init_model(Y, components=((5, 2, 30), ), spl0_comps=0.8, spl0_bg=0.4,
        iterations=10):
    '''
    Rather heuristic initialization procedure.
    
    Y: the data of form [Y] = d x T
    gaussian_blur_memmap : a numpy memory map to a gaussian blur matrix

    components : the number of components (neurons, compartments) we aim for
    with their respective sigma and window size multiplier as tuples
    (n, sigma, wsmult)

    WARNING !! we assume quadratic image shape

    '''
    logger.info('Initializing SMFF model')

    d, T = Y.shape
    R = Y.copy()

    # subtract background ..
    f, b = _nmf_l0(R.T, spl0=spl0_bg, iterations=iterations)
    R  = (R - np.dot(f, b).T).clip(TINY_POSITIVE_NUMBER, np.inf)

    ims = int(np.sqrt(d))

    A, C = [], []

    for num_comps, sg, wsmult in components:

        # calculate the window size
        ws = int(sg*wsmult)
        ws = ws+1 if not np.mod(ws, 2) else ws # make it odd

        logger.info('Finding %s components with sigma %s and window size %s' % (num_comps, sg, ws))
        
        for k in range(num_comps):

            rho = blur_images(R.T.reshape(T,ims,ims),sg).reshape(T,ims**2).T
            rhomax = rho.max(axis=1)
            wcent = np.argmax(rhomax)
            wcent = (np.mod(wcent, ims), int(wcent/ims))
            wmask = window_mask(wcent, (ims, ims), ws).flatten()
            Rw = R[wmask, :]

            H_init = rhomax[wmask].flatten()
            H_init /= H_init.sum() # normalize
            H_init.shape += (1,)
            W_init = np.dot(Rw.T, H_init)

            W, H = _nmf_l0(Rw.T, W_init, spl0=spl0_comps, iterations=iterations)

            ak = np.zeros(R.shape[0])
            ak.flat[wmask] = H
            #ak.flat[wmask] = H/(H[H>0]).std()
            ak = (ak/ak.sum()).clip(TINY_POSITIVE_NUMBER, np.inf) # normalize
            #ak = (ak).clip(TINY_POSITIVE_NUMBER, np.inf) # normalize
            A.append(ak)

            c = np.dot(R.T, ak)
            #W = W.clip(TINY_POSITIVE_NUMBER, np.inf)
            C.append(c/c.sum())

            R[wmask, :] = (R[wmask, :] - np.dot(W, H).T).clip(TINY_POSITIVE_NUMBER, np.inf)


    # initialize background trace + signal
    # get better init estimate for background!!
    #f, b = _nmf_lars_greedy_init(R.T, 'random', k=1, iterations=30, normalize=False, alpha=1.)
    #f, b = _nmf_l0(R.T, alpha=0.01)

    #R  = (R + np.dot(f, b).T).clip(TINY_POSITIVE_NUMBER, np.inf)

    logger.info('Finally calculating background ..')
    f, b = _nmf_l0(R.T, spl0=spl0_bg, iterations=iterations)

    C = np.array(C)  # will be of shape k x T
    A = np.array(A).T # will be of shape d x k

    logger.info('A, C, b and f intitialized.')

    return A, C, b.T, f.T


def blur_images(imagestack, sg, parallel=True):
    bis = Parallel(n_jobs=N_JOBS)(
            delayed(ndimage.gaussian_filter)(imagestack[ii,:,:], sg)
            for ii in range(imagestack.shape[0])
            )
    return np.array(bis)



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
