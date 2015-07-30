

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_spatial_components(A, ncols=5):
    '''
    [A] = d x k
    '''
    d, k = A.shape 
    ims = int(np.sqrt(d))

    nrows = int(np.ceil(k/ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, nrows*3+2))
    for i in range(k):
        if nrows <= 1:
            ax[i % 5].grid(False)
            ax[i % 5].imshow(A[:,i].reshape(ims, ims), cmap='gray')
            ax[i % 5].set_title(str(i))
        else:
            ax[int(i/5)][i % 5].grid(False)
            ax[int(i/5)][i % 5].imshow(A[:,i].reshape(ims, ims), cmap='gray')
            ax[int(i/5)][i % 5].set_title(str(i))

    #tit = plt.suptitle('Spatial Components')
    fig.tight_layout()

    return fig, ax


def plot_temporal_components(C, fs):
    '''
    [C] = k x T
    '''
    k, T = C.shape

    tr = C - np.array([C.mean(axis=1)]*T).T
    tr = tr/np.array([tr.std(axis=1)]*T).T
    
    time = np.linspace(0, T/fs-1/fs, T)
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, k+2))
    yticks = []
    for idx in range(k):
        ytick = idx*-10 
        _ = ax.plot(time, tr[idx]+ytick, lw=1)
        yticks.append(ytick)
    _ = ax.set_yticks(yticks)    
    _ = ax.set_yticklabels([str(i) for i in range(k)])
    _ = ax.set_ylabel('component index')
    _ = ax.set_xlabel('time [s]')

    #tit = plt.suptitle('Temporal Components')
    fig.tight_layout()

    return fig, ax


def plot_correlation_matrix(C, biased=True):
    '''
    [C] = k x T
    '''
    k, T = C.shape
    tr = C - np.array([C.mean(axis=1)]*T).T
    tr = tr/np.array([tr.std(axis=1)]*T).T
    if biased:
        corrmat = np.dot(tr, tr.T)/T
    else:
        corrmat = np.dot(tr, tr.T)/(T-1)

    fig, ax = plt.subplots(figsize=(7, 7))
    _ = sns.heatmap(corrmat, square=True, ax=ax)
    tit = ax.set_title('correlation matrix')
    tit.set_fontsize(14)
    ax.set_xlabel('component index')
    fig.tight_layout()
    return fig, ax


def plot_spectral_components(C, f, fs=15.):
    '''
    '''
    from scipy.signal import welch

    k, T = C.shape

    normC = C - np.array([C.mean(axis=1),]*C.shape[1]).T
    normC /= np.array([normC.std(axis=1),]*normC.shape[1]).T

    normbg = f - f.mean()
    normbg /= normbg.std()

    cbg = np.vstack([normC, normbg])
    ff, ptr = welch(cbg, fs=fs)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.grid(False)
    _ = ax.imshow(ptr,cmap='cubehelix', aspect='auto', extent=[ff[0], ff[-1], k+2, 0], vmin=0)
    _ = ax.set_xlabel('frequency [Hz]')
    _ = ax.set_ylabel('component idx')

    return fig, ax



def browse_stack(imagestack):
    from IPython.html.widgets import interact
    n, y, x = imagestack.shape
    plt.grid(False)
    def view_image(i):
        plt.grid(False)
        plt.imshow(imagestack[i], cmap='gray', interpolation=None)
        plt.title('Frame: %s' % i)
        plt.show()
    interact(view_image, i=(0,n-1))
