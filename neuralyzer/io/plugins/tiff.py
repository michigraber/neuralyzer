
from __future__ import print_function

from neuralyzer.io.loader import LoaderTemplate

FILE_EXTENSIONS = ('tiff', 'tif', )

class Loader(LoaderTemplate):
    
    # TODO : might wanna implement here a general FileHandler

    @staticmethod
    def get_data(filepath):

        try:
            import numpy as np
            from PIL import Image
            return np.array(Image.open(filepath))
        except ImportError:
            pass

        try:
            from tifffile import TiffFile
            mytiff = TiffFile(filepath)
            return mytiff.asarray()
        except:
            raise
