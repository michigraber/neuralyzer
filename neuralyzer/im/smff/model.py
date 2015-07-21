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
from neuralyzer import log

logger = log.get_logger()


TINY_POSITIVE_NUMBER = np.finfo(np.float).tiny 



# SPARSE NON-NEGATIVE MATRIX FACTORIZATION CODE
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
                'fs' : kwargs.get('fs', None),
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
            C, A, b, f = _init_model._init_model(*args, **kwargs)
            self._init_params['C'] = C
            self._init_params['A'] = A
            self._init_params['b'] = b
            self._init_params['f'] = f


    def fit_model(self, Y, max_num_iterations=10, re_init=True, morph_mod=-1,
            temp_update_method='projgrad', **kwargs):
        ''''

        DEPENDING ON THE METHODS CHOSEN YOU CAN PROVIDE DIFFERENT KEYWORD
        ARGUMENTS 

        in any case:
            - temporal_update_method : default='projgrad', 'multiplicative',
              'cvx_foopsie'
            - morph_mod : apply morphological smoothing of spatial components
              every morph_mod step. < 0 leads to no smoothing (default=-1)
            - re_init : initialize model parameters from scratch with stored
              parameters (default=True)
            - spl0 : l0 sparseness measure for spatial components (will be
              removed)
        
        temporal update projgrad (default):
            - tolH (1e-4)
            - maxiter (200)

        cvx_foopsi:
            - noise_range ((0.2, 0.5)) Hz
        '''

        self.logger.info('Fitting SMFF to data Y.')
        self.logger.debug('[Y] = '+str(Y.shape))

        _spl0 = kwargs.pop('spl0', 0.97)

        if re_init:
            self.C = self._init_params['C'].copy()
            self.A = self._init_params['A'].copy()
            self.b = self._init_params['b'].copy()
            self.f = self._init_params['f'].copy()
            self._step = 0
            self.max_num_iterations = max_num_iterations
            self._avg_abs_res = []

        mean_residual = np.abs(self.residual(Y)).mean()
        self.logger.info('avg absolute residual = %s ' % mean_residual)
        self._avg_abs_res.append(mean_residual)

        while not self._stop():
            self.logger.info('iteration %s / %s ' % \
                    (self._step+1, self.max_num_iterations))

            # UPDATE A, b -----------------------------------------------------
            self.A, self.b = SMFF.update_A_b(self.C, self.A, self.b, self.f, Y,
                    spl0=_spl0, logger=self.logger)
            # ROI component post processing
            if not np.mod(self._step+1, morph_mod) and not morph_mod < 0:
                self.logger.info('morphologically closing spatial components.') 
                self.A = _morph_image_components(self.A)
            # UPDATE C, f -----------------------------------------------------
            self.C, self.f = SMFF.update_C_f(self.C, self.A, self.b, self.f, Y,
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

        # TODO : FIX VARIABLE NAMES / NOMENCLATURE FOR BETTER TRANSPARENCY

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

        elif method == 'cvx_foopsi':

            num_rois = A.shape[1]
            resYA = np.dot(Y.T, W) - np.dot(H.T, np.dot(W.T, W))
            nA = (W**2).sum()

            noise = []
            gammas = []
            baseline = []
            c0 = []


        return C, f 


    @staticmethod
    def update_A_b(C, A, b, f, Y, **kwargs):
        '''
        minimize || A ||_1
        subject to : A, b >= 0, ||Y(i,:) - A(i,:)C - b(i)f.T || < \sigma_i * sqrt(T)
        '''
        
        spl0 = kwargs.get('spl0', 0.95)

        logger = kwargs.pop('logger', None)
        if logger: logger.debug('Updating A and b with spl0=%s' % spl0)

        A_, b_ = [], []

        # calculate background data
        if f.ndim == 1:
            f = f[:, np.newaxis].T
        bg = np.dot(b, f)

        # we need to keep all alphas and all hs so that we can cut based on the
        # alpha distribution post hoc -> need a better mechanism here! TODO
        alphas = []
        hs = []

        # LOOP OVER ALL PIXELS (multiple times)
        # TODO : !! parallelize !!

        # 1. model selection loop : shrinkage on A only -----------------------
        for pidx in range(A.shape[0]):
            # i) subtract b from Y
            R = np.array([(Y[pidx] - bg[pidx]),]).clip(TINY_POSITIVE_NUMBER, np.inf).T
            # ii) use strong shrinkage to select components
            ll = nmf.LARS(positive=True)
            ll.fit(C.T, R)
            alphas.append(ll.alphas_)
            hs.append(ll.coef_path_)

        # 2. calculate the cutoff ---------------------------------------------
        alphs = np.concatenate(alphas)
        numcomps = (1.-spl0)*Y.shape[0]*Y.shape[1]
        cutoff = -np.percentile(-alphs, (1.-spl0)*100) 

        # 3. calculate A and b ------------------------------------------------
        H = []
        for n in range(len(hs)):
            relevant_h = hs[n][:, np.where((alphas[n] - cutoff) <= 0)[0][0]]
            comps = np.where(relevant_h)[0]
            Cf = np.vstack([C[comps,:], f])
            y = Y[n]
            y = y[:, np.newaxis]

            # coefficient estimation : selected ROIs including b
            ll = nmf.LARS(positive=True)
            ll.fit(Cf.T, Y[n], return_path=False, alpha_min=0.)

            # append the coeffs
            a = np.zeros(A.shape[1])
            a[comps] = ll.coef_[:-1]
            A_.append(a)
            b_.append(ll.coef_[-1])

        A = np.array(A_)
        b = np.array(b_)
        b = b[:, np.newaxis] # return a 2 dimensional array

        return A, b


    def Y_hat(self):
        ''' The estimated data using the current parameters on the model. '''
        return np.dot(self.A, self.C) + np.dot(self.b, self.f)


    def residual(self, Y):
        ''' '''
        return Y - self.Y_hat()



# -----------------------------------------------------------------------------


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


def _lars(W, v, alpha):
    ll = linear_model.LassoLars(fit_intercept=False, positive=True, alpha=alpha)
    ll.fit(W, v)
    H = np.array(ll.coef_)
    return H, ll.intercept_

def _linear_regression(W, v):
    lr = linear_model.LinearRegression(fit_intercept=False)
    lr.fit(W, v)
    H = np.array(lr.coef_)
    return H
