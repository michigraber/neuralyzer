'''
Calculates an image from a movie where each pixel is the mean correlation of
said pixel with its neighbours in the movie over time.
'''

from __future__ import print_function

import numpy as np

from neuralyzer import log
logger = log.get_logger()

try:
    from sklearn.externals.joblib import Parallel, delayed
    HAS_JOBLIB = True
    N_JOBS = -1
except:
    print('joblib could not be imported. NO PARALLEL JOB EXECUTION!')
    HAS_JOBLIB = False 
    N_JOBS = False


#def save_ci_from_file(infile, outfile, save_ci_data=True, save_cache_infile=False):
    #''' '''
    #from neuralyzer import get_data
    #from matplotlib import pyplot as plt

    #stackdata = get_data(infile, save_cache=save_cache_infile)
    #ci = correlation_image(stackdata)
    
    #fig, ax = plt.subplots()
    #plt.imshow(ci, cmap='cubehelix')
    #fig.imsave(outfile)


def correlation_image(imagestack, njobs=N_JOBS, joblib_tmp_folder='/tmp', joblib_verbosity=0):
    ''' A function that calculates a correlation image from an image stack.
    
    The correlation image is calculated for each pixels by computing 
    the mean correlation of the pixel with its neighbouring pixels.
    
    @arguments: imagestack with dims [N, y, x]
    @returns: correlationimage
    '''
    N, y, x = imagestack.shape
    logger.info('calculating correlation image for stack of dimension (%s, %s, %s)' % (N, y, x))
    npix = y*x
    logger.info('normalizing imagestack ..')
    ims = normalize_imagestack(imagestack)
    ims = ims.reshape(N,npix)
    cim = np.zeros((x,y))

    if HAS_JOBLIB:

        logger.info('calculating correlation image in parallel.')
        cim = Parallel(n_jobs=njobs, temp_folder=joblib_tmp_folder, verbose=joblib_verbosity)(
                delayed(mean_pixel_corr)(ims, pidx, (x,y))
                for pidx in range(npix)
                )
        cim = np.array(cim).reshape(x,y)

    else:

        logger.info('calculating correlation image serially.')
        cim = np.zeros((x,y))
        for idx in range(npix):
           #(xi, yi) = int(idx/x), np.mod(idx, x)
           #nm = mask_neighbours(xi, yi, (x,y)).flatten()
           #cim[xi, yi] = np.dot(ims[:, idx], ims[:, nm]).mean()
            cim[xi, yi] = mean_pixel_corr(ims, idx, (x, y)) 

    return cim


def mean_pixel_corr(ims, pixidx, imshape):
    (xi, yi) = int(pixidx/imshape[0]), np.mod(pixidx, imshape[0])
    nm = mask_neighbours(xi, yi, (imshape[0], imshape[1])).flatten()
    return np.dot(ims[:, pixidx], ims[:, nm]).mean()


def normalize_imagestack(ims):
    ''' Normalize it! What else?
    '''
    ims = ims.copy()
    N, y, x = ims.shape
    ims = ims.reshape(N, y*x)
    ims = ims - np.tile(ims.mean(axis=0), (ims.shape[0], 1))
    ims = ims /np.tile(ims.std(axis=0), (ims.shape[0], 1))
    return ims.reshape(N,y,x)


def get_neighbours_1D(x, size):
    if x >= size:
        raise ValueError('x is bigger than size allows.')
    if x == 0:
        xn = [x, x+1]
    elif x == size-1:
        xn = [x-1, x]
    else:
        xn = [x-1, x, x+1]
    return xn
        

def get_neighbours_2D(x,y, imsize):
    mask = np.zeros(imsize)
    xn = get_neighbours_1D(x, imsize[0])
    yn = get_neighbours_1D(y, imsize[1])
    ns = [(xi, yi) for xi in xn for yi in yn]
    ns.remove((x,y))
    return ns


def mask_neighbours(x, y, imshape):
    mask = np.zeros(imshape, dtype='uint8')
    for (xi, yi) in get_neighbours_2D(x, y, imshape):
        mask[xi, yi] = 1
    return mask.astype('bool')


def mask_random_pixels(N, imshape):
    '''
    N is just the number of times a pixel is drawn.
    If N gets close to the total number of pixels N will definitely not 
    be the number of maskes pixels!
    '''
    mask = np.zeros(imshape, 'uint8').flatten()
    mask[np.random.randint(len(mask)-1, size=(10, 1))] = 1
    return mask.reshape(imshape[0], imshape[1]).astype('bool')
