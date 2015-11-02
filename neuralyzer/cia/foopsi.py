'''
Implementation of the foopsi algorithm from [1] on the basis of the the cvxpy
library.

This code is mostly a pythonization of E. Pnevmatikakis code for matlab
https://github.com/epnev/constrained-foopsi/blob/master/constrained_foopsi.m

.. and makes use of the cvxpy package: https://github.com/cvxgrp/cvxpy 


'''

import numpy as np

from scipy import sparse

from neuralyzer import log

from .smff import noise, autoregressive



def cvx_foopsi(y, w=None, p=2, noise_range=(.25, .5), solver='ECOS',
        logger=log.get_logger()):
    '''
    Computes modeled calcium signal and spike train from measured calcium
    signal.

    Solves the following convex problem  ..

    minimize        G * c
    subject to: 
                    b >= 0,
                    G*c >= 0,
                    y - c - b - c0*gdvec) <= sqrt(T)*sn,
                    c0 >= 0,
    
    .. using cvxpy.

    Calculates sn, c0, G (a matrix with autoregressive model parameters / time
    constants on the (lower) diagonals)

    Based on E. Pnevmatikakis matlab code: 
    https://github.com/epnev/constrained-foopsi/blob/master/constrained_foopsi.m
    '''
    import cvxpy as cvx

    T = len(y)

    sn = noise.sigma_noise_spd_welch(y-y.mean(), 1., noise_range)
    g = autoregressive.AR_est_YW(y-y.mean(), p)
    g = np.insert(-g, 0, 1)
    gd = np.roots(g).max()
    gdvec = gd**np.arange(0, T)

    diags = (np.ones((T, 1))*g[::-1]).T
    G = sparse.spdiags(diags, -np.array(range(len(g)))[::-1], T, T)
    # apparently the scipy sparse package has a bug .. i get an error using it
    # with cvxpy -> therefore we convert G to an array here
    G = G.toarray()

    if w is None or w == []:
        w_ = np.ones((T, 1)).T
    else:
        w_ = np.array(w)
        if np.ndim(w_) == 1:
            w_ = w_[:, np.newaxis]
        if w_.shape == (T, 1):
            w_ = w_.T
        elif w_.shape == (1, T): pass
        else:
            raise ValueError('w has to be of shape (1, T), (T, 1) or (T,)')


    # CVXPY 
    c = cvx.Variable(T)
    b = cvx.Variable()
    c0 = cvx.Variable()

    baseline_lower_bound = max((0., min(y)))

    objective = cvx.Minimize(w_*(G*c))
    constraints = [
            b >= baseline_lower_bound,
            G*c >= 0,
            cvx.norm(y - c - c0*gdvec - b) <= np.sqrt(T)*sn,
            c0 >= 0,
            ]

    problem = cvx.Problem(objective, constraints)
    _v = problem.solve(solver=solver)

    if problem.status == 'infeasible':
        #if logger is not None:
            #logger.info('cvxpy problem solver status: {0}'.format(problem.status))
            #logger.debug('calculating new sigma_noise ..')

        objective = cvx.Minimize(cvx.norm(y - c - c0*gdvec - b))
        constraints = [
                b >= baseline_lower_bound,
                G*c >= 0,
                c0 >= 0,
                ]

        problem_ = cvx.Problem(objective, constraints)
        _v = problem_.solve(solver=solver)
        sn = _v/np.sqrt(T)

        #if logger is not None:
            #logger.info('cvxpy problem solver status: {0}'.format(problem_.status))

    elif problem.status in ['unbounded', ]:
        if logger is not None:
            logger.warning('cvxpy problem solver status: {0}'.format(problem.status))
            #logger.debug('sigma_noise = {0}'.format(sn))
        raise CVXFoopsiError('cvxpy problem solver status is "{0}"'.format(problem.status))
    else:
        pass
        #if logger is not None:
            #logger.info('cvxpy problem solver status: {0}'.format(problem.status))

    #if logger is not None: logger.debug('sigma_noise = {0}'.format(sn))

    c_ = np.array(c.value).squeeze()
    c__ = c_ + c0.value*gdvec
    events = np.dot(G, c__)
    events[events < 10.**-10] = 0.
    events[:p] = 0. # set the first p (ar order) time points to zero due to inaccuracies

    b_ = b.value
    
    # we're only interested in the real part of the solutions found 
    return c__.real, events.real, b_.real, sn.real, g.real


class CVXFoopsiError(Exception):
    pass
