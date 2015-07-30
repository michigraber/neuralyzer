'''
An implementation of Calcium Signal extraction, demixing and ROI estimation
along the model and algorithm described in 

> A structured matrix factorization framework for large scale calcium imaging
> data analysis.

by Pnevmatikakis et al. 2014

'''

import numpy as np

from sklearn import linear_model 
from sklearn.decomposition.nmf import _nls_subproblem

from neuralyzer.im import nmf

from neuralyzer.im.smff import noise

from neuralyzer import log
logger = log.get_logger()


try:
    from joblib import Parallel, delayed
    N_JOBS = -1
except:
    print 'joblib could not be imported. NO PARALLEL JOB EXECUTION!'
    N_JOBS = None 


TINY_POSITIVE_NUMBER = np.finfo(np.float).tiny 



# STRUCTURED NON-NEGATIVE MATRIX FACTORIZATION CODE
# -----------------------------------------------------------------------------

class SMFF(object):
    ''' A non-negative matrix factorization approach for calcium imaging data.
    
    A calcium imaging dataset Y, [Y] = d x T, is factorized according to

    Y = AC + bf.T

    with [A] = d x k, [C] = k x T, [b] = d x 1, [f] = T x 1
    '''

    def __init__(self, *args, **kwargs):
        self.logger = kwargs.pop('logger', logger)
        self._init_params = {} 
        # check for initialization parameters
        for ini in ('A', 'C', 'f', 'b'):
            self._init_params[ini] = kwargs.pop(ini, None)

        self.params = {
                'fs' : kwargs.get('fs', 15.),
                'noise_range' : kwargs.get('noise_range', (0.25, 0.5)), # in fs units
                }


    def _stop(self):
        '''
        simple interation number based stop criterion for now
        '''
        # alternatively we could calculate the residual and estimate whether it
        # behaves noise style ..
        return self._step >= self.max_num_iterations


    def init_model(self, *args, **kwargs):
        if kwargs.get('random', False):
            k = kwargs.get('k', None)
            d = kwargs.get('d', None)
            T = kwargs.get('T', None)
            self._init_params['C'] = np.random.rand(k, T)
            self._init_params['A'] = np.random.rand(d, k)
            self._init_params['b'] = np.random.rand(d, 1)
            self._init_params['f'] = np.random.rand(1, T)

        else:
            from . import _init_model
            A, C, b, f = _init_model.greedy(*args, **kwargs)
            self._init_params['C'] = C
            self._init_params['A'] = A
            self._init_params['b'] = b
            self._init_params['f'] = f


    def fit_model(self, Y, max_num_iterations=10, re_init=True, morph_mod=-1,
            temp_update_method='projgrad', njobs=N_JOBS, **kwargs):
        ''''

        DEPENDING ON THE METHODS CHOSEN YOU CAN PROVIDE DIFFERENT KEYWORD
        ARGUMENTS 

        in any case:
            - temporal_update_method : default='projgrad', 'multiplicative',
              'constrained_foopsi'
            - morph_mod : apply morphological smoothing of spatial components
              every morph_mod step. < 0 leads to no smoothing (default=-1)
            - re_init : initialize model parameters from scratch with stored
              parameters (default=True)
        
        temporal update projgrad (default):
            - tolH (1e-4)
            - maxiter (200)

        constrained_foopsi:
            - noise_range ((0.25, 0.5), in fs units)
        '''

        self.logger.info('Fitting SMFF to data Y.')
        self.logger.debug('njobs = %s' % njobs)

        if re_init:
            self.logger.debug('Copying initial model values ..')
            self.C_ = self._init_params['C'].copy()
            self.A_ = self._init_params['A'].copy()
            self.b_ = self._init_params['b'].copy()
            self.f_ = self._init_params['f'].copy()
            self._step = 0
            self.max_num_iterations = max_num_iterations
            self._avg_abs_res = []

        mean_residual = np.abs(self.residual(Y)).mean()
        self.logger.info('avg absolute residual = %s ' % mean_residual)
        self._avg_abs_res.append(mean_residual)

        if temp_update_method in ('projgrad', 'constrained_foopsi',):
            self.logger.debug('calculating noise level for all pixels.')
            self._pixel_noise = noise.sigma_noise_spd_welch(Y, 1., self.params['noise_range'])

        while not self._stop():
            self.logger.info('iteration %s / %s ' % \
                    (self._step+1, self.max_num_iterations))

            # UPDATE A, b -----------------------------------------------------
            self.A_, self.b_ = SMFF.update_A_b(self.C_, self.A_, self.b_, self.f_, Y,
                    self._pixel_noise, logger=self.logger, njobs=njobs)
            # ROI component post processing
            if not np.mod(self._step+1, morph_mod) and not morph_mod < 0:
                self.logger.info('morphologically closing spatial components.') 
                self.A_ = _morph_image_components(self.A_)

            # TODO : threshold spatial components
            if kwargs.get('threshold', False):
                pass

            # UPDATE C, f -----------------------------------------------------
            self.C_, self.f_ = SMFF.update_C_f(self.C_, self.A_, self.b_, self.f_, Y,
                    method=temp_update_method, logger=self.logger, **kwargs)

            # ROI Merging -----------------------------------------------------
            # TODO

            # RESIDUAL CALCULATION --------------------------------------------
            mean_residual = np.abs(self.residual(Y)).mean()
            self.logger.info('avg absolute residual = %s ' % mean_residual)
            self._avg_abs_res.append(mean_residual)

            # update counter after completion of entire step
            self._step += 1

    
    @staticmethod
    def update_C_f(C, A, b, f, Y, **kwargs):
        '''
        '''
        method = kwargs.pop('method', 'projgrad')

        logger = kwargs.pop('logger', None)
        if logger: logger.debug('Updating C and f with method "%s".' % method)

        # adding the background data to the matrices
        W = np.hstack((A, b))
        H = np.vstack((C, f))

        if method == 'projgrad':
            tolH = kwargs.get('tolH', 1e-4)
            maxiter = kwargs.get('maxiter', 200)
            # using the modified scikit-learn project gradient _nls_subproblem
            # update.
            H_, grad, n_iter = _nls_subproblem(Y, W, H, tolH, maxiter)
            # rearrangement of output 
            C = H_[:-1, :]
            f = H_[-1, :]
            f = f[:, np.newaxis].T

        elif method == 'multiplicative':
            # since this method is based on multiplication and division we need
            # to ensure that we will not get infinity errors
            W = W.clip(TINY_POSITIVE_NUMBER, np.inf)
            H = H.clip(TINY_POSITIVE_NUMBER, np.inf)
            # we call the multiplicative update from our Morup NMF implementation
            W_ = nmf.NMF_L0.update_W(Y.T, W.T, H.T)
            # rearrangement of output 
            C = W_[:, :-1].T
            f = W_[:, -1].T 
            f = f[:, np.newaxis].T

        elif method == 'constrained_foopsi':

            # along the cvx_foopsi part of the Pnevmatikakis code on github:
            # https://github.com/epnev/constrained-foopsi/blob/master/constrained_foopsi.m

            num_rois = A.shape[1]
            resYA = np.dot(Y.T, W) - np.dot(H.T, np.dot(W.T, W))
            nA = (W**2).sum()

            gammas = []
            baseline = []
            c0 = []

        return C, f 


    @staticmethod
    def update_A_b(C, A, b, f, Y, noise, **kwargs):
        '''
        minimize || A ||_1
        subject to : A, b >= 0, ||Y(i,:) - A(i,:)C - b(i)f.T || < \sigma_i * sqrt(T)
        '''
        logger = kwargs.pop('logger', None)
        if logger: logger.debug('Updating A and b')

        njobs = kwargs.pop('njobs', N_JOBS)

        d, T = Y.shape
        H = np.vstack((C, f))
        
        if njobs is None:
            A_ = []
            for pidx, sn in enumerate(noise):
                A_.append(nmf.do_lars_fit(H.T, Y[pidx], alpha=sn*np.sqrt(T)))

        elif type(njobs) == int:
                A_ = Parallel(n_jobs=njobs)(
                        delayed(nmf.do_lars_fit)(H.T, Y[pidx], alpha=noise[pidx]*np.sqrt(T))
                        for pidx in range(len(noise)))
        else:
            raise ValueError('njobs of improper type. Can only be an int or None.')

        A_ = np.array(A_)
        A = A_[:,:-1]
        A /= np.linalg.norm(A, axis=0)[np.newaxis, :]
        b = np.dot((Y - np.dot(A, C)), f.T/norm(f))
        b /= np.linalg.norm(b)

        return A, b


    def Y_hat(self):
        ''' The estimated data using the current parameters on the model. '''
        return np.dot(self.A_, self.C_) + np.dot(self.b_, self.f_)


    def residual(self, Y):
        ''' '''
        return Y - self.Y_hat()



# UTILITIES
# -----------------------------------------------------------------------------

def norm(x):
    ''' euclidian norm for 1d vector '''
    return np.sqrt(np.dot(x.squeeze(),x.squeeze()))

def _morph_image_components(H):
    from skimage import morphology
    m, k = H.shape
    w = int(np.sqrt(m))
    imshape = (w,w)
    for i in range(k):
        H[:, i] = morph_close_component(H[:,i], imshape) 
    return H

def morph_close_component(a, imshape):
    from skimage import morphology
    amorph = morphology.closing(a.reshape(imshape[0], imshape[1]))
    return amorph.flatten()
