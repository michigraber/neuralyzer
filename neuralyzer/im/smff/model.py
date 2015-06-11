'''
An implementation of Calcium Signal extraction, demixing and ROI estimation
along the model and algorithm described in 

> A structured matrix factorization framework for large scale calcium imaging
> data analysis.

by Pnevmatikakis et al. 2014

'''

import numpy as np

from sklearn import linear_model 

from neuralyzer.im import nmf
from neuralyzer import log

logger = log.get_logger()


TINY_POSITIVE_NUMBER = np.finfo(np.float).tiny 



# SPARSE NON-NEGATIVE MATRIX FACTORIZATION CODE
# -----------------------------------------------------------------------------

class SMFF(object):
    ''' An non-negative matrix factorization approach for calcium imaging data.

    
    A calcium imaging dataset Y, [Y] = p x T, is factorized according to

    Y = AC + bf.T

    with [A] = p x k, [C] = k x T, [b] = p x 1, [f] = T x 1
    '''

    def __init__(self, *args, **kwargs):
        self.logger = kwargs.pop('logger', logger)

        self.max_iter = kwargs.pop('max_iter', 100)
        self._step = 0

        self.alpha_sel = kwargs.pop('alpha_sel', 5.0)
        self.alpha_fit = kwargs.pop('alpha_fit', 0.01)

        self._model_params = ('A', 'C', 'f', 'b')
        self._init_params = {} 
        # check for initialization parameters
        for ini in self.model_params:
            self._init_params[ini] = kwargs.pop(ini, None)
            if ini in kwargs:
                self._init_params[ini] = kwargs[ini]

    def _stop(self):
        '''
        simple interation number based stop criterion for now
        '''

        # alternatively we could calculate the residual and estimate whether it
        # behaves noise style ..

        return self._step >= self.max_iter


    @property
    def model_params(self):
        return self._model_params


    def init_model(self, *args, **kwargs):
        from . import _init_model
        C, A, b, f = _init_model._init_model(*args, **kwargs)
        self._init_params['C'] = C
        self._init_params['A'] = A
        self._init_params['b'] = b
        self._init_params['f'] = f


    def fit_model(self, Y, **kwargs):

        self.C = self._init_params['C'].copy()
        self.A = self._init_params['A'].copy()
        self.b = self._init_params['b'].copy()
        self.f = self._init_params['f'].copy()

        self.logger.info('Fitting SMFF to data Y.')
        self.logger.debug('[Y] = '+str(Y.shape))
    
        while not self._stop():
            self.logger.info('iteration %s / %s ' % (self._step+1, self.max_iter))
            self._step += 1
            # UPDATE A, b
            self.A, self.b = self._update_A_b(self.C, self.A, self.b, self.f, Y)
            # ROI component post processing
            self.A = _morph_image_components(self.A)
            # UPDATE C, f
            self.C, self.f = self._update_C_f(self.C, self.A, self.b, self.f, Y)


    def _update_C_f(self, C, A, b, f, Y):
        '''
        minimize \sum (1 G c_j)
        subject to Gc_j >= 0, ||Y(i,:) - A(i,:)C - b(i)f.T || < \sigma_i * sqrt(T)

        NOOOOOOT! (for now)
        '''
        # recast the variables
        V = Y.T
        H = np.vstack([A.T, b])
        W = np.vstack([C, f.T]).T
        # multiplicative update
        W_ = nmf.NMF_L0._update_W(V, H, W)
        
        C = W_[:, :-1].T
        f = W_[:, -1].T 

        return C, f 


    def _update_A_b(self, C, A, b, f, Y, spl0=0.95):
        '''
        minimize || A ||_1
        subject to : A, b >= 0, ||Y(i,:) - A(i,:)C - b(i)f.T || < \sigma_i * sqrt(T)
        '''
        A_, b_ = [], []

        # calculate background data
        if f.ndim == 1:
            f = f[:, np.newaxis]
        bg = np.dot(f, b).T

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

        # 3. calculate A and b
        H = []
        for n in range(len(hs)):
            relevant_h = hs[n][:, np.where((alphas[n] - cutoff) <= 0)[0][0]]
            comps = np.where(relevant_h)[0]
            Cf = np.vstack([C[comps,:], f.T])
            y = Y[n]
            y = y[:, np.newaxis]

            # coefficient estimation : selected ROIs including b
            ll = nmf.LARS(positive=True)
            ll.fit(Cf.T, Y[n], return_path=False, alpha_min=self.alpha_fit)

            # append the coeffs
            a = np.zeros(A.shape[1])
            a[comps] = ll.coef_[:-1]
            A_.append(a)
            b_.append(ll.coef_[-1])

        A = np.array(A_)
        b = np.array(b_)
        b = b[:, np.newaxis] # return a 2 dimensional array

        return A, b.T


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
