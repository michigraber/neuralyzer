'''
Functions to estimate the coefficients on an autoregressive (AR) model,
see below.



Code shamelessly copied from various submodules of the nipy project:
github.com/nipy/nitime/


LICENCE ::

Copyright (c) 2006-2011, NIPY Developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NIPY Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np

from scipy import linalg
from scipy import fftpack
from scipy.signal import signaltools



def AR_est_YW(x, order, rxx=None, sigma_v=False):
    r"""Determine the autoregressive (AR) model of a random process x using
    the Yule Walker equations. The AR model takes this convention:
    .. math::
      x(n) = a(1)x(n-1) + a(2)x(n-2) + \dots + a(p)x(n-p) + e(n)
    where e(n) is a zero-mean white noise process with variance sig_sq,
    and p is the order of the AR model. This method returns the a_i and
    sigma
    The orthogonality property of minimum mean square error estimates
    states that
    .. math::
      E\{e(n)x^{*}(n-k)\} = 0 \quad 1\leq k\leq p
    Inserting the definition of the error signal into the equations above
    yields the Yule Walker system of equations:
    .. math::
      R_{xx}(k) = \sum_{i=1}^{p}a(i)R_{xx}(k-i) \quad1\leq k\leq p
    Similarly, the variance of the error process is
    .. math::
      E\{e(n)e^{*}(n)\}   = E\{e(n)x^{*}(n)\} = R_{xx}(0)-\sum_{i=1}^{p}a(i)R^{*}(i)
    Parameters
    ----------
    x : ndarray
        The sampled autoregressive random process
    order : int
        The order p of the AR system
    rxx : ndarray (optional)
        An optional, possibly unbiased estimate of the autocorrelation of x
    Returns
    -------
    ak : The estimated AR coefficients 
    sig_sq : The innovations variance (if sigma_v=True)
    """
    if rxx is not None and type(rxx) == np.ndarray:
        r_m = rxx[:order + 1]
    else:
        r_m = autocorr(x)[:order + 1]

    Tm = linalg.toeplitz(r_m[:order])
    y = r_m[1:]
    ak = linalg.solve(Tm, y)
    if sigma_v:
        sigma_v = r_m[0].real - np.dot(r_m[1:].conj(), ak).real
        return ak, sigma_v
    else:
        return ak


def autocorr(x, **kwargs):
    """Returns the autocorrelation of signal s at all lags.
    Parameters
    ----------
    x : ndarray
    axis : time axis
    all_lags : {True/False}
       whether to return all nonzero lags, or to clip the length of r_xy
       to be the length of x and y. If False, then the zero lag correlation
       is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2
    Notes
    -----
    Adheres to the definition
    .. math::
    R_{xx}[k]=E\{X[n+k]X^{*}[n]\}
    where X is a discrete, stationary (ergodic) random process
    """
    # do same computation as autocovariance,
    # but without subtracting the mean
    kwargs['debias'] = False
    return autocov(x, **kwargs)


def autocov(x, **kwargs):
    """Returns the autocovariance of signal s at all lags.
    Parameters
    ----------
    x : ndarray
    axis : time axis
    all_lags : {True/False}
       whether to return all nonzero lags, or to clip the length of r_xy
       to be the length of x and y. If False, then the zero lag correlation
       is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2
    Returns
    -------
    cxx : ndarray
       The autocovariance function
    Notes
    -----
    Adheres to the definition
    .. math::
    C_{xx}[k]=E\{(X[n+k]-E\{X\})(X[n]-E\{X\})^{*}\}
    where X is a discrete, stationary (ergodic) random process
    """
    # only remove the mean once, if needed
    debias = kwargs.pop('debias', True)
    axis = kwargs.get('axis', -1)
    if debias:
        x = remove_bias(x, axis)
    kwargs['debias'] = False
    return crosscov(x, x, **kwargs)


def crosscov(x, y, axis=-1, all_lags=False, debias=True, normalize=True):
    """Returns the crosscovariance sequence between two ndarrays.
    This is performed by calling fftconvolve on x, y[::-1]
    Parameters
    ----------
    x : ndarray
    y : ndarray
    axis : time axis
    all_lags : {True/False}
       whether to return all nonzero lags, or to clip the length of s_xy
       to be the length of x and y. If False, then the zero lag covariance
       is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2
    debias : {True/False}
       Always removes an estimate of the mean along the axis, unless
       told not to (eg X and Y are known zero-mean)
    Returns
    -------
    cxy : ndarray
       The crosscovariance function
    Notes
    -----
    cross covariance of processes x and y is defined as
    .. math::
    C_{xy}[k]=E\{(X(n+k)-E\{X\})(Y(n)-E\{Y\})^{*}\}
    where X and Y are discrete, stationary (or ergodic) random processes
    Also note that this routine is the workhorse for all auto/cross/cov/corr
    functions.
    """
    if x.shape[axis] != y.shape[axis]:
        raise ValueError(
            'crosscov() only works on same-length sequences for now'
            )
    if debias:
        x = remove_bias(x, axis)
        y = remove_bias(y, axis)
    slicing = [slice(d) for d in x.shape]
    slicing[axis] = slice(None, None, -1)
    cxy = fftconvolve(x, y[tuple(slicing)].conj(), axis=axis, mode='full')
    N = x.shape[axis]
    if normalize:
        cxy /= N
    if all_lags:
        return cxy
    slicing[axis] = slice(N - 1, 2 * N - 1)
    return cxy[tuple(slicing)]


def remove_bias(x, axis):
    "Subtracts an estimate of the mean from signal x at axis"
    padded_slice = [slice(d) for d in x.shape]
    padded_slice[axis] = np.newaxis
    mn = np.mean(x, axis=axis)
    return x - mn[tuple(padded_slice)]


def fftconvolve(in1, in2, mode="full", axis=None):
    """ Convolve two N-dimensional arrays using FFT. See convolve.
    This is a fix of scipy.signal.fftconvolve, adding an axis argument and
    importing locally the stuff only needed for this function
    """
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    complex_result = (np.issubdtype(in1.dtype, np.complex) or
                      np.issubdtype(in2.dtype, np.complex))

    if axis is None:
        size = s1 + s2 - 1
        fslice = tuple([slice(0, int(sz)) for sz in size])
    else:
        equal_shapes = s1 == s2
        # allow equal_shapes[axis] to be False
        equal_shapes[axis] = True
        assert equal_shapes.all(), 'Shape mismatch on non-convolving axes'
        size = s1[axis] + s2[axis] - 1
        fslice = [slice(l) for l in s1]
        fslice[axis] = slice(0, int(size))
        fslice = tuple(fslice)

    # Always use 2**n-sized FFT
    fsize = 2 ** int(np.ceil(np.log2(size)))
    if axis is None:
        IN1 = fftpack.fftn(in1, fsize)
        IN1 *= fftpack.fftn(in2, fsize)
        ret = fftpack.ifftn(IN1)[fslice].copy()
    else:
        IN1 = fftpack.fft(in1, fsize, axis=axis)
        IN1 *= fftpack.fft(in2, fsize, axis=axis)
        ret = fftpack.ifft(IN1, axis=axis)[fslice].copy()
    del IN1
    if not complex_result:
        ret = ret.real
    if mode == "full":
        return ret
    elif mode == "same":
        if np.product(s1, axis=0) > np.product(s2, axis=0):
            osize = s1
        else:
            osize = s2
        return signaltools._centered(ret, osize)
    elif mode == "valid":
        return signaltools._centered(ret, abs(s2 - s1) + 1)
