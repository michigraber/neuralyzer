'''
Noise estimation. 
'''
import numpy as np

def sigma_noise_spd_welch(y, fs, noise_range, method='expmeanlog'):
    '''
    Estimating the noise level by a spectral power density (welch algorithm)
    approach.

    ARGUMENTS
    `````````
    y : signal, [y] = N x T
    fs : sampling rate
    noise_range : noise range (a, b) in Hz, implemented as interval [a, b)

    If you choose fs to be one you can indicate your noise_range in units of
    the sampling rate. Be aware that only [0, 0.5) is feasible.

    method : ('expmeanlog', 'mean', 'median',)
    '''

    from scipy.signal import welch

    if len(y.shape) == 1:
        T = y.shape[0]
        dim = 1
    elif len(y.shape) == 2:
        N, T = y.shape
        dim = 2
    else:
        raise ValueError('y has dimension %s. Only 1 or 2 are allowed.' % len(y.shape))

    f, pxx = welch(y, nperseg=np.round(T/8), noverlap=0, nfft=1000, fs=fs)
    # implementing the range as [a, b)
    fmask = (f >= noise_range[0])*(f < noise_range[1])

    if dim == 1:
        if method == 'expmeanlog':
            return np.sqrt(np.exp(np.mean(np.log(pxx[fmask]))))
        elif method == 'mean':
            return np.sqrt(np.mean(pxx[fmask]))
        elif method == 'median':
            return np.sqrt(np.median(pxx[fmask]))
        else:
            NotImplementedError('The method chosen is not available.')

    # TODO : parallelization might be better than using welch multidim
    elif dim == 2:
        if method == 'expmeanlog':
            return np.sqrt(np.exp(np.mean(np.log(pxx[:, fmask]), axis=1)))
        elif method == 'mean':
            return np.sqrt(np.mean(pxx[:, fmask], axis=1))
        elif method == 'median':
            return np.sqrt(np.median(pxx[:, fmask], axis=1))
        else:
            NotImplementedError('The method chosen is not available.')
