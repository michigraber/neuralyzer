
def get_data(filename, cache_data=True):
    ''' A function to load tiff data files and its faster loading caches.

    In case the file has been loaded previously and caching was enabled
    (default=True) then a way faster loading cache file has been saved. This
    function will load this one automatically.

    :Args:
        - filename : specify either an absolute or a relative path to a file
          you would like to load.

    :Kwargs:
        - cache_data=True : if no cache is available, create a faster loading
          cache file.
     '''
    from ..io import data_handler
    return data_handler.DataHandler().get_data(filename, cache_data=cache_data)
