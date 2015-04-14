
from neuralyzer.io.loader import LoaderTemplate

FILE_EXTENSIONS = ('txt', )

class Loader(LoaderTemplate):
    
    @staticmethod
    def get_data(filepath,):
        try:
            return get_txt_data(filepath)
        except:
            raise 


# reading files
# -----------------------------------------------------------------------------

def get_txt_data(filepath, splitter='\r\n', dtype=float):
    import numpy as np
    with open(filepath) as fid:
        content = fid.read()
    return np.array([dtype(el) for el in content.split(splitter) if el])
