
from neuralyzer.io.loader import LoaderTemplate

FILE_EXTENSIONS = ('tiff', 'tif', )

class Loader(LoaderTemplate):
    
    # TODO : might wanna implement here a general FileHandler

    @staticmethod
    def get_data(filepath):
        try:
            from tifffile import TiffFile
            mytiff = TiffFile(filepath)
            return mytiff.asarray()
        except:
            raise
