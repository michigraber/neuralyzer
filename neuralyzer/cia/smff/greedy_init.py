'''
@Model Initialization Code.

Since NMF has not a unique solution, the outcome of the optimization procedure
depends on the initialization procedure. Random initialization is typically not
optimal.

Here we implemented a fast heuristic model initialization procedure along the
original reference by Pnevmatikakis et al. 2014.
'''

from __future__ import print_function

import numpy as np 
from scipy import ndimage

try:
    from sklearn.externals.joblib import Parallel, delayed
    N_JOBS = -1
    JOBLIB_TMP_FOLDER = '/tmp'
except:
    print('joblib could not be imported. NO PARALLEL JOB EXECUTION!')
    N_JOBS = None 

from .. import nmf 

from neuralyzer import log
logger = log.get_logger()

TINY_POSITIVE_NUMBER = np.finfo(np.float).tiny 


def greedy(Y, components=((5, 2, 30), ), spl0_comps=0.1, iterations=5, njobs=N_JOBS):
    '''
    Rather heuristic initialization procedure.
    
    Y: the data of form [Y] = d x T

    components : the number of components (neurons, compartments, windowsize
    multiplier) we aim for with their respective sigma and window size
    multiplier as tuples (n, sigma, wsmult)

    WARNING !! we assume quadratic image shape


    TODO : noise constrained nmf -> in order to not have to return the entire path

    '''
    logger.info('Initializing SMFF model with greedy algorithm.')

    d, T = Y.shape
    ims = int(np.sqrt(d))

    logger.debug('copying data') 
    R = Y.copy()

    # subtract 'background' ..
    logger.info('subtracting background')
    b = np.mean(R, axis=1)
    b /= np.linalg.norm(b)
    b = b[:, np.newaxis]
    f = np.dot(R.T, b).T
    R  = (R - np.dot(b, f))

    A, C = [], []

    sigmagauss = 0

    for num_comps, sg, wsmult in components:

        if sigmagauss != sg:
            sigmagauss = sg
            # calculate the window size
            ws = int(sigmagauss*wsmult)
            ws = ws+1 if not np.mod(ws, 2) else ws # make it odd
            newblur = True
        else:
            newblur = False

        logger.info('Finding %s components with sigma %s and window size %s' % (num_comps, sigmagauss, ws))
                
        if newblur:
            rho = blur_images(R.T.reshape(T,ims,ims), sigmagauss,
                    njobs=njobs).reshape(T,ims**2).T
        
        for k in range(num_comps):
            logger.debug('component %s / %s' % (k+1, num_comps))

            # find maximum coords in max of gauss convolved imagestack (rho)
            rhomax = rho.max(axis=1)
            wcent = np.argmax(rhomax)
            # 2D image coordinates
            wcent = (np.mod(wcent, ims), int(wcent/ims))
            wmask = window_mask(wcent, (ims, ims), ws)
            mask_w = wmask.sum(axis=0).max()
            mask_h = wmask.sum(axis=1).max()
            wmask = wmask.flatten()

            Rw = R[wmask, :]
            H_init = rhomax[wmask].flatten()
            H_init /= np.linalg.norm(H_init) # normalize
            H_init.shape += (1,)
            W_init = np.dot(Rw.T, H_init)

            W, H = _nmf_l0(Rw.T, W_init=W_init, H_init=H_init.T,
                    spl0=spl0_comps, iterations=iterations)

            # creating the full size component
            ak = TINY_POSITIVE_NUMBER*np.ones(d)
            ak.flat[wmask] = H
            ak[ak < 2*TINY_POSITIVE_NUMBER] = 0. # thresholding at 2 epsilon
            ak /= np.linalg.norm(ak) # normalize
            A.append(ak)

            c = np.dot(R.T, ak)
            #W = W.clip(TINY_POSITIVE_NUMBER, np.inf)
            #C.append(c/c.sum())
            C.append(c)

            #R[wmask, :] = (R[wmask, :] - np.dot(W, H).T).clip(TINY_POSITIVE_NUMBER, np.inf)
            # subtract fitted patch from 'data' R and current rho
            fitpatch = np.dot(W,H)
            R[wmask, :] = (R[wmask, :] - fitpatch.T)
            
            blurpatch = blur_images(fitpatch.reshape(T, mask_w, mask_h),
                    sigmagauss, njobs=njobs)

            rho[wmask, :] -= blurpatch.reshape(T, mask_w*mask_h).T


    C = np.array(C)  # will be of shape k x T
    A = np.array(A).T # will be of shape d x k

    logger.info('Finally calculating background ..')
    Yres =  Y - np.dot(A, C)
    b = Yres.mean(axis=1)
    b /= np.linalg.norm(b)
    b = b[:, np.newaxis]
    f = np.dot(Yres.T, b).T

    logger.info('A, C, b and f initialization completed.')

    return A, C, b, f



@log.log_profiling
def blur_images(imagestack, sg, njobs=N_JOBS, joblib_tmp_folder=JOBLIB_TMP_FOLDER):
    ''' 2D Gaussian filter on all images of imagestack. '''

    if njobs is None:
        bis = []
        for ii in range(imagestack.shape[0]):
            bis.append(ndimage.gaussian_filter(imagestack[ii,:,:], sg))

    elif type(njobs) == int:
        # TODO : what about the temp folder now?
        #bis = Parallel(n_jobs=njobs, temp_folder=joblib_tmp_folder)(
        bis = Parallel(n_jobs=njobs)(
                delayed(ndimage.gaussian_filter)(imagestack[ii,:,:], sg)
                for ii in range(imagestack.shape[0])
                )

    return np.array(bis)


@log.log_profiling
def _nmf_l0(V, W_init=None, H_init=None, spl0=0.6, iterations=10, k=1):
    '''
    '''
    nmfl0 = nmf.NMF_L0(spl0=spl0, iterations=iterations)
    nmfl0.fit(V, W_init=W_init, H_init=H_init, k=k)
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
