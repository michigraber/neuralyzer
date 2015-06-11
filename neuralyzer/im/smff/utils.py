'''
'''

# TODO
# -----------------------------------------------------------------------------
# - write function that saves memmap file of gaussian blur matrix for arbitrary
#   sizes
# - 
# 

def gauss_kernel(size, var=None):    
    if type(size) == int:
        size = (size, size)
    else:
        raise ValueError('We only provide symmetric kernels with sizes specified by an int.')
    if not np.mod(size[0],2) or not np.mod(size[1],2):
        raise ValueError('The size of the Kernel has to be odd.')
    s = (int(size[0]/2.), int(size[0]/2.))
    if var is None: v = s
    elif type(var) == int: v = (var, var)
    else: v = var
    x, y = np.mgrid[-s[0]:s[0]+1, -s[1]:s[1]+1]
    g = np.exp(-(x**2/(2*float(v[0]))+y**2/(2*float(v[1]))))
    return g / g.sum()


def gaussian_blur_image(imagesize, kernelsize, kernelvariance, position):
    pos = position
    ims = imagesize
    ks = kernelsize
    kernel = _gauss_kernel(ks, kernelvariance)
    # generate a padded image
    im = np.zeros((ims[0]+ks-1, ims[1]+ks-1))
    im[pos[0]:pos[0]+ks, pos[1]:pos[1]+ks] = kernel
    # remove padding
    im = im[int(ks/2):-int(ks/2), int(ks/2):-int(ks/2)]
    return im


def gaussian_blur_matrix(imagesize, kernelsize, kernelvariance):
    dim = imagesize[0]*imagesize[1]
    D = np.zeros((dim, dim))
    for idx in range(dim):
        pos = (int(idx/imagesize[0]), np.mod(idx, imagesize[0]))
        D[:,idx] = _gaussian_blur_image(
                imagesize, kernelsize, kernelvariance, pos
                ).flatten()
    return D




"""

SPARSE GAUSSIAN BLUR MATRIX : 

- > deprecated but structurally interesting


def gaussian_blur_matrix_sparse(dim, kernelsize, kernelvariance):
    '''
    
    D is a symmetric matrix!

    tested with : plt.imshow(D.toarray()[:100,:100])
    '''
    from scipy import sparse as sps
    kernel = _gauss_kernel(kernelsize, kernelvariance).flatten()
    kernel = kernel.flatten()
    s = int(len(kernel)/2)
    data = []
    offsets = []
    for i in range(len(kernel)):
        data.append(np.ones(dim)*kernel[i])
        offsets.append(s-i)
    D = sps.dia_matrix((np.array(data), np.array(offsets)), shape=(dim, dim))
    return D
"""


