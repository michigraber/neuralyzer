
import os
import re
import json


def get_data(filename, cache_data=True, **kwargs):
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
    return data_handler.DataHandler().get_data(filename, cache_data=cache_data, **kwargs)


def get_meta_file(path, meta_file_name='data_meta.json', dirretracts=3):
    ''' Search for data_meta.json file in the parent directories '''
    if os.path.isfile(path):
        trunk, _ = os.path.split(path)
    else:
        trunk = path
    for _ in range(dirretracts):
        meta_file = os.path.join(trunk, meta_file_name)
        if os.path.exists(meta_file):
            return meta_file
        trunk, tail = os.path.split(trunk)


def get_meta_info(path, dirrectracts=3, idselector=r'.*/(?P<id>\w*)_.*.\w*$'):
    ''' Search for data_meta.json and return its content + recid '''
    metafile = get_meta_file(path, dirretracts=dirrectracts)
    if metafile is None:
        return
    with open(metafile, 'r') as jfi:
        metadict = json.load(jfi)
    return metadict


def get_rec_id(filename, idselector=r'.*/(?P<id>\w*)_.*.\w*$'):
    ''' Read-out recording id from filename. '''
    recidmatch = re.match(idselector, path)
    if recidmatch is None:
        raise ValueError('Could not identify pattern.')        
    recid = recidmatch.groupdict()['id']
    return recid
