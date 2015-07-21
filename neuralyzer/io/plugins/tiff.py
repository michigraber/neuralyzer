
from __future__ import print_function

from neuralyzer.io.loader import LoaderTemplate

FILE_EXTENSIONS = ('tiff', 'tif', )

DEFAULT_LIBRARY = 'PIL'

class Loader(LoaderTemplate):
    
    @staticmethod
    def get_data(filepath, library=DEFAULT_LIBRARY):

        if library == 'PIL':
            try:
                return get_data_PIL(filepath)
            except:
                raise 

        elif library == 'tifffile':
            try:
                return get_data_tifffile(filepath)
            except:
                raise

        else:
            raise ValueError('No load method implemented for library %s' % library)


# reading files using PIL
# -----------------------------------------------------------------------------

PILMode2NPdtype = {
        'I;16': 'uint16',
        'I;16B': 'uint16',
        }

def get_data_PIL(filepath):
    import numpy as np
    from PIL import Image, ImageSequence

    img = Image.open(filepath)
    try:
        dtype = PILMode2NPdtype[img.mode]
    except KeyError:
        raise NotImplementedError(('The handling of tif files with PIL'
                ' mode "%s" is currently not supported.') % img.mode)

    framedata = [np.array(frame)
            for frame in ImageSequence.Iterator(img)]
    image = np.array(framedata)
    img.close()
    return image 


# reading files using tifffile
# -----------------------------------------------------------------------------

def get_data_tifffile(filepath):
    ''' reads a tifffile using the tifffile library '''
    from tifffile import TiffFile
    mytiff = TiffFile(filepath)
    return mytiff.asarray()
