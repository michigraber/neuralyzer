

def rescale_stack(imstack, scale, **kwargs):

    from skimage import transform
    n, y, x = imstack.shape
    rsstack = []
    for ii in range(n):
        rsstack.append(transform.rescale(imstack[ii,:,:], scale, **kwargs))
    return np.array(rsstack)
