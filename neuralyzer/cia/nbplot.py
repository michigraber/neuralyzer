
from __future__ import print_function

import os

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


def browse_components(A, C, fs, G=None, Y=None, S=None, center=True, cmap='cubehelix', outpath=None):

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
        print('size of segments: ', np.bincount(skimage.measure.label(iso).ravel())[1:])
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
            events = (S[i] > S[i].std()*10.)
            print('num events: ', events.sum())
            print('mean events size: ', C[i,events].mean()/C[i].std())

        ax2.set_xlabel('time [s]')

        if G is not None:
            print('G: ', G[i])
        
        if outpath is not None:
            fig.savefig(os.path.join(outpath, 'component_{0}.png'.format(i)))
        plt.show()
    interact(view_image, i=(0,k-1))





C = [
        [219,255,241],
        [218,255,239],
        [218,255,238],
        [217,254,236],
        [216,253,234],
        [216,253,233],
        [215,252,231],
        [215,252,229],
        [214,251,228],
        [213,250,226],
        [213,250,224],
        [212,249,223],
        [212,249,221],
        [211,248,219],
        [211,247,218],
        [210,247,216],
        [209,246,214],
        [209,246,213],
        [208,245,211],
        [208,244,209],
        [207,244,208],
        [206,243,206],
        [206,243,204],
        [205,242,203],
        [205,242,201],
        [204,241,199],
        [204,240,198],
        [203,240,196],
        [202,239,194],
        [202,239,193],
        [201,238,191],
        [201,237,189],
        [200,237,188],
        [199,236,186],
        [199,236,184],
        [198,235,183],
        [198,234,181],
        [197,234,179],
        [197,233,178],
        [196,233,176],
        [195,232,174],
        [195,231,173],
        [194,231,171],
        [194,230,169],
        [193,230,167],
        [193,229,166],
        [192,228,164],
        [191,228,162],
        [191,227,161],
        [190,227,159],
        [190,226,157],
        [189,225,156],
        [188,225,154],
        [188,224,152],
        [187,224,151],
        [187,223,149],
        [186,223,147],
        [186,222,146],
        [185,221,144],
        [184,221,142],
        [184,220,141],
        [183,220,139],
        [183,219,137],
        [182,218,136],
        [181,218,135],
        [179,216,135],
        [178,214,135],
        [176,213,135],
        [174,211,135],
        [173,210,136],
        [171,208,136],
        [169,206,136],
        [168,205,136],
        [166,203,136],
        [164,202,137],
        [163,200,137],
        [161,199,137],
        [159,197,137],
        [158,195,138],
        [156,194,138],
        [154,192,138],
        [153,191,138],
        [151,189,138],
        [149,188,139],
        [147,186,139],
        [146,184,139],
        [144,183,139],
        [142,181,140],
        [141,180,140],
        [139,178,140],
        [137,177,140],
        [136,175,140],
        [134,173,141],
        [132,172,141],
        [131,170,141],
        [129,169,141],
        [127,167,141],
        [126,166,142],
        [124,164,142],
        [122,162,142],
        [121,161,142],
        [119,159,143],
        [117,158,143],
        [115,156,143],
        [114,154,143],
        [112,153,143],
        [110,151,144],
        [109,150,144],
        [107,148,144],
        [105,147,144],
        [104,145,145],
        [102,143,145],
        [100,142,145],
        [99,140,145],
        [97,139,145],
        [95,137,146],
        [94,136,146],
        [92,134,146],
        [90,132,146],
        [89,131,147],
        [87,129,147],
        [85,128,147],
        [83,126,147],
        [82,125,147],
        [80,123,148],
        [78,121,148],
        [77,120,148],
        [75,118,148],
        [74,117,148],
        [73,116,147],
        [72,115,146],
        [71,114,145],
        [70,113,144],
        [70,113,143],
        [69,112,143],
        [68,111,142],
        [67,110,141],
        [66,109,140],
        [66,108,139],
        [65,107,138],
        [64,106,137],
        [63,106,136],
        [62,105,135],
        [61,104,134],
        [61,103,134],
        [60,102,133],
        [59,101,132],
        [58,100,131],
        [57,99,130],
        [56,98,129],
        [56,98,128],
        [55,97,127],
        [54,96,126],
        [53,95,125],
        [52,94,125],
        [52,93,124],
        [51,92,123],
        [50,91,122],
        [49,90,121],
        [48,90,120],
        [47,89,119],
        [47,88,118],
        [46,87,117],
        [45,86,117],
        [44,85,116],
        [43,84,115],
        [42,83,114],
        [42,83,113],
        [41,82,112],
        [40,81,111],
        [39,80,110],
        [38,79,109],
        [38,78,108],
        [37,77,108],
        [36,76,107],
        [35,75,106],
        [34,75,105],
        [33,74,104],
        [33,73,103],
        [32,72,102],
        [31,71,101],
        [30,70,100],
        [29,69,99],
        [28,68,99],
        [28,67,98],
        [27,67,97],
        [26,66,96],
        [25,65,95],
        [24,64,94],
        [23,63,93],
        [23,62,92],
        [22,61,91],
        [21,60,90],
        [21,59,89],
        [21,58,87],
        [20,57,86],
        [20,56,84],
        [20,56,83],
        [19,55,82],
        [19,54,80],
        [19,53,79],
        [18,52,77],
        [18,51,76],
        [18,50,74],
        [17,49,73],
        [17,48,72],
        [17,47,70],
        [16,46,69],
        [16,45,67],
        [16,44,66],
        [15,43,64],
        [15,42,63],
        [15,41,62],
        [14,40,60],
        [14,39,59],
        [14,38,57],
        [13,37,56],
        [13,36,54],
        [13,35,53],
        [12,34,51],
        [12,34,50],
        [12,33,49],
        [11,32,47],
        [11,31,46],
        [11,30,44],
        [10,29,43],
        [10,28,41],
        [10,27,40],
        [9,26,39],
        [9,25,37],
        [8,24,36],
        [8,23,34],
        [8,22,33],
        [7,21,31],
        [7,20,30],
        [7,19,29],
        [6,18,27],
        [6,17,26],
        [6,16,24],
        [5,15,23],
        [5,14,21],
        [5,13,20],
        [4,12,19],
        [4,11,17],
        [4,11,16],
        [3,10,14],
        [3,9,13],
        [3,8,11],
        [2,7,10],
        [2,6,9],
        [2,5,7],
        [1,4,6],
        [1,3,4],
        [1,2,3],
        [0,1,1],
        [0,0,0],
    ]
