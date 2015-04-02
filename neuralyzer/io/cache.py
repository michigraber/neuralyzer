'''
'''

import os

DEFAULT_CACHE_METHOD = 'hdf5'



# DATA CACHE
# -----------------------------------------------------------------------------

def get_cache_filename(filepath, cache_method=DEFAULT_CACHE_METHOD):
    return filepath+'.'+cache_method


class CacheError(Exception):
    pass


class DataCache(object):
    '''
    '''

    def __init__(self, filepath, cache_method=DEFAULT_CACHE_METHOD):
        self.filepath = os.path.abspath(filepath)
        self._cache_method = cache_method
        self._cache_filepath = get_cache_filename(filepath, cache_method=cache_method)
        self.cache_handler = _get_cache_handler(cache_method, self._cache_filepath)

    @property
    def cache_method(self):
        return self._cache_method
    
    @property
    def cache_filepath(self):
        return self._cache_filepath

    @property
    def is_cached(self):
        return os.path.exists(self.cache_filepath)

    def get_cached_data(self):
        # check whether file is cached
        if not self.is_cached:
            raise CacheError('File is not cached.')
        return self.cache_handler.load_data()

    def save_data_cache(self, data):
        self.cache_handler.save_data(data)
        

# CACHE HANDLER
# -----------------------------------------------------------------------------

class CacheHandler(object):
    '''
    An abstract API defining class only.
    '''

    # TODO : make explicitly abstract using abc

    def __init__(self, cachefile):
        self.cachefile = cachefile

    def load_data(self):
        raise NotImplementedError()

    def save_data(self, data):
        raise NotImplementedError()


class H5PY_CacheHandler(CacheHandler):

    DEFAULT_DATASETNAME = 'data'

    def __init__(self, *args, **kwargs):
        super(H5PY_CacheHandler, self).__init__(*args, **kwargs)

    def load_data(self, datasetname=DEFAULT_DATASETNAME):
        try:
            import h5py
            with h5py.File(self.cachefile, 'r') as h5file:
                data = h5file[datasetname][:]
        except ImportError:
            CacheError('h5py is not installed on your computer, but required.')
        except:
            raise
        return data

    def save_data(self, data, datasetname=DEFAULT_DATASETNAME):
        try:
            import h5py
            with h5py.File(self.cachefile, 'w') as h5file:
                data = h5file.create_dataset(datasetname, data=data)
        except ImportError:
            CacheError('h5py is not installed on your computer, but required.')
        except:
            raise


CACHE_HANDLER_REGISTRY = {
        'hdf5' : H5PY_CacheHandler,
        }

def _get_cache_handler(method, cachefile):
    if not method in CACHE_HANDLER_REGISTRY:
        raise NotImplementedError('There is no handler registered for this method.')
    return CACHE_HANDLER_REGISTRY[method](cachefile)
