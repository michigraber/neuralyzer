'''
A structured approach to calcium signal extraction, demixing and ROI estimation
from calcium imaging data along the model and algorithm described by
Pnevmatikakis et al. in [1].

Author : Michael H. Graber <michigraber@gmail.com>
License : MIT License

References
----------
[1] Pnevmatikakis et al. (2014) A structured matrix factorization framework
for large scale calcium imaging data analysis. arXiv preprint
arXiv:1409.2903.  http://arxiv.org/abs/1409.2903

'''

from __future__ import print_function

import numpy as np
from scipy import sparse

from sklearn.decomposition.nmf import _nls_subproblem

from neuralyzer.cia import nmf, foopsi
from . import noise

from neuralyzer import log
logger = log.get_logger()


try:
    from sklearn.externals.joblib import Parallel, delayed
    N_JOBS = -1
except:
    print('joblib could not be imported. NO PARALLEL JOB EXECUTION!')
    N_JOBS = None 


TINY_POSITIVE_NUMBER = np.finfo(np.float).tiny 



# STRUCTURED NON-NEGATIVE MATRIX FACTORIZATION
# -----------------------------------------------------------------------------

class SMFF(object):
    ''' Non-negative matrix factorization for calcium imaging data.
    
    A calcium imaging dataset Y, [Y] = d x T, is factorized according to

    Y = AC + bf.T

    with [A] = d x k, [C] = k x T, [b] = d x 1, [f] = T x 1

    A being the spatial components, C the calcium signal, b the background
    component and f the background signal.


    Model Parameters / Keyword Arguments
    ------------------------------------

    temporal_update_method : ('projgrad' (default) | 'multiplicative' | 'foopsi')
        The method/algorithm used to update the temporal components.
        foopsi includes a model-based estimation of spike / event times.

    noise_range : (default=(0.3, 0.5))
        Spectral range for the estimation of the signal noise in units of the
        temporal sampling rate.

    njobs : (default=-1)
        If joblib is installed computation can be distributed to multiple
        processors. -1 distributes to all processors.

    iterations : (default=3)
        Number of block coordinate descent iterations for entire model.

    filt_it : (default=0)
        Execute spatial filtering on the spatial components every filt_it
        iteration. 0 (or any smaller number) leads to no filtering.

    Depending on your temporal_update_method of choice you can additionally set
    the following parameters:

    'projgrad'
    ``````````
    tolH : (default=1e-4)

    maxiter : (default=2000)

    'foopsi'
    ````````
    foopsi_autoreg_p : (default=3)
        Order of the autoregressive model for the calcium signal.

    foopsi_bcd_its : (default=5)
        Block coordinate descent iterations for the foopsi temporal component
        update.


    References
    ----------
    [1] Pnevmatikakis et al. (2014) A structured matrix factorization framework
    for large scale calcium imaging data analysis. arXiv preprint
    arXiv:1409.2903.  http://arxiv.org/abs/1409.2903

    '''

    def __init__(self, **kwargs):
        self.logger = kwargs.pop('logger', logger)

        self._model_init = {} 
        self._step = 0
        self.avg_abs_res_ = []


        # SET UP THE PARAMETERS

        # default model parameters
        self.params = {
                'temporal_update_method' : 'foopsi',
                'noise_range' : (0.25, 0.5), # in fs units
                'njobs' : -1,
                'iterations': 3,
                'filt_it': 0,
                }
        self.params.update(**kwargs)

        # conditional parameters
        if self.params['temporal_update_method'] == 'foopsi':
            self.params.update({
                'foopsi_bcd_its' : kwargs.get('foopsi_bcd_its', 5),
                'foopsi_autoreg_p' : kwargs.get('foopsi_autoreg_p', 3),
                })
        elif self.params['temporal_update_method'] == 'projgrad':
            self.params.update({
                'projgrad_tolH' : kwargs.get('projgrad_tolH', 1e-4),
                'projgrad_maxiter' : kwargs.get('maxiter', 2000),
                })


        paramstring = ', '.join("{!s}={!r}".format(k,v) for (k,v) in self.params.items())
        self.logger.info('SMFF({0})'.format(paramstring))
        

    def init_model(self, *args, **kwargs):
        ''' Initialize the model values.  '''

        # check for initialization parameters
        for ini in ('A', 'C', 'f', 'b'):
            self._model_init[ini] = kwargs.pop(ini, None)

        if not any([v is None for v in self._model_init.values()]):
            self.logger.info('Model entirely initialized with kwargs.')
            return

        if kwargs.get('random', False):
            k = kwargs.get('k', None)
            d = kwargs.get('d', None)
            T = kwargs.get('T', None)
            for k, v in self._model_init.items():
                if v is None:
                    self._model_init[k] = {
                            'C' : np.random.rand(k, T),
                            'A' : np.random.rand(d, k),
                            'b' : np.random.rand(d, 1),
                            'f' : np.random.rand(1, T),
                            }[k]
        else:
            from . import greedy_init 
            self.params['greedy_init_params'] = kwargs
            A, C, b, f = greedy_init.greedy(*args, **kwargs)
            self._model_init['C'] = C
            self._model_init['A'] = A
            self._model_init['b'] = b
            self._model_init['f'] = f


    def fit_model(self, Y, copy_init=True):
        ''' Fit the data to the model with the specified parameters.  '''
        self.logger.info('Fitting SMFF model to data Y. [Y] = (%s, %s)' % Y.shape)
        self._tap_model_init(copy=copy_init)

        mean_residual = np.abs(self.calculate_residual(Y)).mean()
        self.logger.info('avg absolute residual = %s ' % mean_residual)
        self.avg_abs_res_.append(mean_residual)

        while not self._stop():
            self.logger.info('iteration %s / %s ' % \
                    (self._step+1, self.params['iterations']))
            self._do_bcd_step(Y, **self.params)

    
    def _tap_model_init(self, copy=True):
        ''' Copy or refer model coefficients to _model_init '''
        if copy:
            self.logger.info('Copying initial model values ..')
            self.C_ = self._model_init['C'].copy()
            self.A_ = self._model_init['A'].copy()
            self.b_ = self._model_init['b'].copy()
            self.f_ = self._model_init['f'].copy()
        else:
            self.C_ = self._model_init['C']
            self.A_ = self._model_init['A']
            self.b_ = self._model_init['b']
            self.f_ = self._model_init['f']

    
    def _stop(self):
        ''' Simple interation number based stop criterion for now. '''
        return self._step >= self.params['iterations']


    def _do_bcd_step(self, Y, **params):
        ''' Executes a single block gradient descent iteration step on the
        whole model.

        Model parameters can be overwritten using kwargs here.
        '''
        params.update(self.params)

        # we need to compute the pixelwise noise only once
        if not hasattr(self, 'pixel_noise_'):
            self.logger.info('calculating noise level for all pixels.')
            self.pixel_noise_ = noise.sigma_noise_spd_welch(Y, 1., params['noise_range'])

        # UPDATE A, b ---------------------------------------------------------
        self.A_, self.b_ = SMFF.update_A_b(
            self.C_, self.A_, self.b_, self.f_, Y, self.pixel_noise_,
            njobs=params['njobs'], logger=self.logger
            )

        # throw away components containing nan
        remove_mask = np.isnan(self.A_).any(axis=0)
        self.A_ = self.A_[:, ~remove_mask]
        self.C_ = self.C_[~remove_mask]

        # ROI component post processing
        if not ( np.mod(self._step+1, params['filt_it']) or params['filt_it'] < 1 ):
            self.logger.info('filter spatial components.') 
            self.A_ = filter_spatial_components(self.A_, disk_size=2)

        # TODO : threshold spatial components ??
        #if params.get('threshold_A', False):
            #pass

        # UPDATE C, f ---------------------------------------------------------
        self.C_, self.f_, self.S_, self.G_ = SMFF.update_C_f(
                self.C_, self.A_, self.b_, self.f_, Y,
                logger=self.logger, **params)

        # drop inactive components 
        if self.S_ is not None:
            remove_mask = ~self.S_.any(axis=1)
            if any(remove_mask):
                self.logger.info('Removing inactive components {0}'.format(
                    np.where(remove_mask)[0]))
            self.A_ = self.A_[:, ~remove_mask]
            self.C_ = self.C_[~remove_mask]

        # ROI Merging ---------------------------------------------------------
        # TODO

        # RESIDUAL CALCULATION ------------------------------------------------
        mean_residual = np.abs(self.calculate_residual(Y)).mean()
        self.logger.info('avg absolute residual = %s ' % mean_residual)
        self.avg_abs_res_.append(mean_residual)

        self._step += 1

    
    @staticmethod
    def update_C_f(C, A, b, f, Y, **kwargs):
        ''' Update the temporal components C and f.  '''

        logger = kwargs.pop('logger', None)
        params = kwargs
        if logger:
            logger.info('Updating C and f with method "{0}".'.format(
                params['temporal_update_method']))

        # adding the background data to the matrices
        W = np.hstack((A, b))
        H = np.vstack((C, f))

        if params['temporal_update_method'] == 'foopsi':
            N, T = H.shape
            # projection of the residual onto the spatial components
            resYA = np.dot((Y - np.dot(W, H)).T, W)
            H_ = np.zeros((N, T))
            S_ = np.zeros((N-1, T))
            G_ = np.zeros((N-1, params['foopsi_autoreg_p']))
            # foopsi block coordinate descent iterations
            for bcd_it in range(params['foopsi_bcd_its']):
                # randomly permute component indices 
                for ii in np.random.permutation(range(N)):
                    # all regular components
                    if ii < N-1:
                        resYA[:,ii] = resYA[:,ii] + H[ii]
                        c_, spks_, b_, sn_, g_ = foopsi.cvx_foopsi(resYA[:, ii],
                                noise_range=params['noise_range'],
                                p=params['foopsi_autoreg_p'])
                        H_[ii, :] = (c_ + b_).squeeze()
                        resYA[:,ii] = resYA[:,ii] - H_[ii, :]
                        S_[ii, :] = spks_.squeeze()
                        G_[ii, :] = g_[1:].squeeze()
                    # the background
                    else:
                        resYA[:,ii] = resYA[:,ii] + H[ii]
                        H_[ii, :] = resYA[:, ii].clip(0, np.inf)
                        resYA[:,ii] = resYA[:,ii] - H_[ii, :]

            C = H_[:N-1,:]
            f = H_[N-1,:]
            f = f[:, np.newaxis].T

        elif params['temporal_update_method'] == 'projgrad':
            # using the scikit-learn project gradient _nls_subproblem update
            H_, grad, n_iter = _nls_subproblem(Y, W, H,
                    params['projgrad_tolH'], params['projgrad_maxiter'])
            # rearrangement of output 
            C = H_[:-1, :]
            f = H_[-1, :]
            f = f[:, np.newaxis].T
            S_, G_ = None, None

        elif params['temporal_update_method'] == 'multiplicative':
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
            S_, G_ = None, None

        else:
            raise ValueError('temporal_update_method %s not available.' % \
                    params['temporal_update_method'])

        return C, f, S_, G_ 


    @staticmethod
    def update_A_b(C, A, b, f, Y, pixel_noise, **kwargs):
        ''' Update the spatial components A and the background b.  '''

        logger = kwargs.get('logger', None)
        if logger: logger.info('Updating A and b')

        d, T = Y.shape
        H = np.vstack((C, f))
        
        njobs = kwargs.get('njobs', N_JOBS)
        if njobs is None:
            A_ = []
            for pidx, sn in enumerate(noise):
                A_.append(nmf.do_lars_fit(H.T, Y[pidx], alpha=sn*np.sqrt(T)))

        elif type(njobs) == int:
                sqrtT = np.sqrt(T)
                A_ = Parallel(n_jobs=njobs)(
                        delayed(nmf.do_lars_fit)(
                            H.T, Y[pidx], alpha=pixel_noise[pidx]*sqrtT
                            )
                        for pidx in range(len(pixel_noise)))
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


    def calculate_residual(self, Y):
        ''' '''
        return Y - self.Y_hat()




# UTILITIES
# -----------------------------------------------------------------------------

def norm(x):
    ''' euclidian norm for 1d vector '''
    return np.sqrt(np.dot(x.squeeze(),x.squeeze()))

def _morph_image_components(H):
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

def morph_erode_component(a, imshape, disk_size=1):
    from skimage import morphology
    amorph = morphology.erosion(
            a.reshape(imshape[0], imshape[1]),
            morphology.disk(disk_size))
    return amorph.flatten()

def filter_spatial_components(H, filter_method='erosion', **kwargs):
    ''' '''
    w = int(np.sqrt(H.shape[0])) # nastily assuming square images here
    imshape = (w,w)
    if filter_method == 'erosion':
        for i, imflat in enumerate(H.T):
            H[:, i] = morph_erode_component(imflat, imshape, **kwargs)
    elif filter_method == 'closing':
        for i, imflat in enumerate(H.T):
            H[:, i] = morph_erode_component(imflat, imshape, **kwargs)
    H /= np.linalg.norm(H) # normalize
    return H
