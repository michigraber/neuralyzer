'''
Utility functions for the SMF module.
'''

import os
import json

from neuralyzer.io.data_handler import DataHandler


def save_smf_parameters(smf, paramfile='smf_parameters.json', overwrite=False):
    ''' Saves the parameters used in a run to a file.
    '''
    path, filename = os.path.split(paramfile)
    dh = DataHandler(path)
    _, ftype = os.path.splitext(filename)
    filepath = dh.place_file(filename)
    if os.path.exists(filepath) and not overwrite:
        raise IOError('File already exists but overwrite=False.')
    if ftype == '.json':
        with open(dh.place_file(filename), 'w') as fid:
            fid.write(json.dumps({ 'SMF PARAMETERS': smf.params }, indent=2))
    else:
        raise ValueError('Filetype not recognized: %s' % ftype)

