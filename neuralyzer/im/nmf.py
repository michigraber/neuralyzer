'''


'''

import numpy as np

from neuralyzer.log import get_logger


def nmf_cvxpy(A, k, max_iter=30):
    '''
    An alternating convex optimization Ansatz.

    This implementation is rather slow ..
    '''

    import cvxpy as cvx
    
    m, n = A.shape

    # Initialize Y randomly.
    Y = np.random.rand(m, k)

    # Perform alternating minimization.
    residual = np.zeros(max_iter)
    for iter_num in range(1, 1+max_iter):
        # At the beginning of an iteration, X and Y are NumPy
        # array types, NOT CVXPY variables.

        # For odd iterations, treat Y constant, optimize over X.
        if iter_num % 2 == 1:
            X = cvx.Variable(k, n)
            constraint = [X >= 0]
        # For even iterations, treat X constant, optimize over Y.
        else:
            Y = cvx.Variable(m, k)
            constraint = [Y >= 0]
        
        # Solve the problem.
        obj = cvx.Minimize(cvx.norm(A - Y*X, 'fro'))
        prob = cvx.Problem(obj, constraint)
        prob.solve(solver=cvx.SCS)

        if prob.status != cvx.OPTIMAL:
            raise Exception("Solver did not converge!")
        
        print 'Iteration {}, residual norm {}'.format(iter_num, prob.value)
        residual[iter_num-1] = prob.value

        # Convert variable to NumPy array constant for next iteration.
        if iter_num % 2 == 1:
            X = X.value
        else:
            Y = Y.value

    return X, Y



def nmf_lars(V, k, H_init=None, max_iter=30, morph=True,
        log=False, **kwargs):
    '''
    
    V = WH : V, W, H >= 0

    V.shape = (m, n)
    W.shape = (m, k)
    H.shape = (k, n)

    '''

    from sklearn.linear_model import LassoLars 

    logger = get_logger()
    
    m, n = V.shape

    # Initialize W randomly.
    if H_init is not None:
        H = H_init
    else:
        H = np.random.rand(n, k)
    

    # Perform 'alternating' minimization.
    for iter_num in range(max_iter):

        if log:
            logger.info('iteration : %s / %s' % (iter_num+1, max_iter))

        lB = LassoLars(positive=True, max_iter=200, alpha=0.5, normalize=True, **kwargs)
        lB.fit(H, V.T)
        W = lB.coef_

        lA = LassoLars(positive=True, max_iter=200, alpha=0.05, normalize=True, **kwargs)
        lA.fit(W, V)
        H = lA.coef_

        if morph:
            H = _morph_image_components(H)

    return W, H



def _morph_image_components(H, imshape=(128,128)):
    from skimage import morphology
    m, k = H.shape
    for i in range(k):
        H[:, i] = morph_close_component(H[:,i], imshape) 
    return H


def median_filter_component(a, imshape):
    from skimage import filters, morphology
    # FIXME : disk size choice is arbitrary right now ..
    return filters.median(a.reshape(*imshape)/a.max(), morphology.disk(5)).flatten()


def morph_close_component(a, imshape):
    from skimage import morphology
    amorph = morphology.closing(a.reshape(imshape[0], imshape[1]))
    return amorph.flatten()




# -----------------------------------------------------------------------------
# Approximate L0 constrained Non-negative Matrix and Tensor Factorization 
# -----------------------------------------------------------------------------

from sklearn.linear_model import LassoLars


class NMF_L0(object):
    ''' An implementation of non-negative matrix factorization according to [1]

    Solves the equation     V = WH 
    for                     V, W, H >= 0
    with                    [V] = N x M, [W] = N x k, [H] = k x M 

    .. in an iterative, multiplicatve approach under Lars Lasso regularization
    of the number of active elements in H.

    References:
    [1] M. Morup, K. H. Madsen, L. K. Hansen, Approximate L0 constrained
    Non-negative Matrix and Tensor Factorization, IEEE 2008
    '''

    def __init__(self, alpha=1.0, iterations=200):
        self.alpha = alpha
        # TODO: iterations will have to be replaced by a stop criterion,
        # based on a the residual and max_iterations!
        self.iterations = iterations 


    def fit(self, V, H_init=None, W_init=None, k=None):
        self.v_shape = V.shape
        if W_init is None:
            self._W = np.random.rand(self.v_shape[0], k)
        else:
            self._W = W_init
        if H_init is None:
            self._H = np.random.rand(k, self.v_shape[1])
        else:
            self._H = H_init
        for i in range(self.iterations):
            # FIXME : here i would like to change the order but i often run
            # into numerical out of bounds errors ..
            self._H = NMF_L0._update_H(self._W, V, alpha=self.alpha)
            self._W = NMF_L0._update_W(V, self._H, self._W)
            #self._H = NMF_L0._update_H(self._W, V, alpha=self.alpha)


    @staticmethod
    def _update_H(W, V, alpha=1.0):
        '''
        !!! WARNING : requires tweaked scikit-learn LassoLars implementation
        that allows non-negativity, ie positivity, constraint on H.
        '''
        H = []
        ll = LassoLars(positive=True, alpha=alpha)
        # TODO: can be paralellized along V dim 1, not zero!
        for n in range(V.shape[1]):
            ll.fit(W, V[:,n])
            H.append(ll.coef_)
        H = np.array(H).T
        return H


    @staticmethod
    def _update_W(V, H, W):
        n, m = V.shape
        n, k = W.shape
        W = np.multiply(W,
                np.multiply(
                    np.dot(V, H.T) + 
                        np.dot(W,
                            np.diag(
                                np.dot(
                                    np.ones((1, n)),
                                    np.multiply(np.dot(W, np.dot(H, H.T)), W)
                                ).flatten()
                            )
                        ),
                    1./(np.dot(W, np.dot(H, H.T)) + 
                        np.dot(
                            W, 
                            np.diag(
                                np.dot(np.ones((1, n)),
                                    np.multiply(np.dot(V, H.T), W)
                                    ).flatten()
                            )
                        )
                       )
                )
            )
        return W

