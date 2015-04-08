
from __future__ import print_function

from neuralyzer.io.loader import LoaderTemplate

DEFAULT_IMAGE_SIZE = (512, 512)

class Loader(LoaderTemplate):
    
    @staticmethod
    def get_data(filepath, imsize=DEFAULT_IMAGE_SIZE):
        return read_raw_file_image_stack()

# reading raw files
# -----------------------------------------------------------------------------

def read_raw_file_image_stack(filepath, imsize, datatype='<u2'):
    arr = read_raw_file(filepath, datatype=datatype)
    imstack = imarr.reshape((imarr.shape[0]/(imsize[0]*imsize[1]), imsize[0], imsize[1]))
    return imstack


def read_raw_file(filepath, datatype='')
    '''
    datatype '<u2' means little-endian unsigned integer 16-bit (ie 2 bytes)

    for specification of array data types, see:
    http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
    '''
    import numpy as np
    with open(filepath,  'r') as fid:
        arr = np.memmap(fid, dtype=endianness+datatype, mode='r')
    return np.array(arr)
