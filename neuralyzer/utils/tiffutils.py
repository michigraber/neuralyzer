'''
Utility functions for tiff file handling.
'''

def imarray_from_files_in_directory(directory, pat=r'\w*.tif'):
    '''
    Returns an numpy array with the data in all files in directory matching a
    regexp pattern.
    '''

    import numpy as np
    from PIL import Image
    from ..io import data_handler

    dh = data_handler.DataHandler(root_path=directory)
    imagelist = dh.get_listdir(pat=pat)
    if not any(imagelist):
        raise ValueError(('Specified directory contains no files matching the'
            ' regexp pattern.'))
    imagelist = sorted([i[0] for i in imagelist])

    firstdata = dh.get_data(imagelist[0], cache_data=False)
    if len(firstdata.shape) == 2:
	shape = firstdata.shape
    elif len(firstdata.shape) == 3:
        shape = firstdata.shape[1:]
    NImages = len(imagelist)
    imarray = np.zeros((NImages, shape[0], shape[1]),
            firstdata.dtype)
    
    for imind in range(NImages):
        imarray[imind,:,:]= dh.get_data(imagelist[imind], cache_data=False)

    return imarray, imagelist


def merge_stack_from_files_in_directory(directory, filename='stack.tif',
        pat=r'\w*.tif'):
    '''
    Merges all files in directory matching the regexp a pattern into a single
    tiff stack.
    '''
    imarray, _ = imarray_from_files_in_directory(directory, pat=pat)
    write_tifffile(filename)

def write_tifffile(filename, imarray):
    import tifffile
    tifffile.imsave(filename, imarray)
