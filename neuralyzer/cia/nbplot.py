

import numpy as np

import seaborn as sns
sns.set(style='whitegrid')

from matplotlib import pyplot as plt


def plot_spatial_components(A, ncols=5):
    '''
    [A] = d x k
    '''
    d, k = A.shape 
    ims = int(np.sqrt(d))

    nrows = int(np.ceil(k/float(ncols)))
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


def plot_correlation_matrix(C, biased=True, figsize=(8,8)):
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

    fig, ax = plt.subplots(figsize=figsize)
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
    def view_image(i):
        plt.grid(False)
        plt.imshow(imagestack[i], cmap='gray', interpolation=None)
        plt.title('Frame: %s' % i)
        plt.show()
    interact(view_image, i=(0,n-1))


def browse_components(A, C, fs, Y=None, S=None, center=True, cmap='cubehelix'):

    from IPython.html.widgets import interact
    from matplotlib import gridspec
    from component_properties import isolate_component
    import skimage

    d, k = A.shape
    k, T = C.shape
    ims = np.sqrt(d)
    if center:
        tr = C - np.array([C.mean(axis=1)]*T).T
        tr = tr/np.array([tr.max(axis=1)]*T).T
    else:
        tr = C
    if Y is not None:
        P = np.dot(Y.T, A)
        P -= np.array([P.mean(axis=0)]*T)
        P /= np.array([P.max(axis=0)]*T)
        #P *= np.array([tr.max(axis=1)]*T)
    if S is not None:
        S /= np.array([S.max(axis=1)]*T).T
        #S *= np.array([tr.max(axis=1)]*T).T
    time = np.linspace(0, T/fs-1/fs, T)

    def view_image(i):
        #fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))
        # image

        fig = plt.figure(figsize=(14, 14))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
        ax0 = plt.subplot(gs[0, 0])
        ax0.grid(False)
        ax0.imshow(A[:,i].reshape(ims, ims), cmap=cmap, interpolation=None)
        ax0.set_title('Frame: %s' % i)
        ax1 = plt.subplot(gs[0, 1])
        ax1.grid(False)
        iso = isolate_component(A[:, i].reshape(ims, ims))
        ax1.imshow(iso, cmap='gray')
        print 'size of segments: ', np.bincount(skimage.measure.label(iso).ravel())[1:]
        # trace
        ax2 = plt.subplot(gs[1, :])
        ax2.plot(time, tr[i])
        if Y is not None:
            isop = Y[iso.flatten(),:].mean(axis=0)
            dff = isop/isop.mean()
            ax2.plot(time, dff-2., 'gray', lw=1)
            ax2.plot(time, P[:,i]-2.)
        if S is not None:
            ax2.plot(time, (dff.max()-1.)*S[i]-1., '-r', lw=0.5)
        plt.show()
    interact(view_image, i=(0,k-1))
