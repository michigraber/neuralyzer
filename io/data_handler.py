'''
'''

import os
import importlib

def io_plugins():
    # might not be the most beautiful solution to the plugin problem
    ps = {}
    pluginpath = os.path.join(os.path.dirname(__file__), 'plugins')
    pluginfiles = [f for f in os.listdir(pluginpath)
            if not (not f.endswith('.py') or f.startswith('_'))]
    for pf in pluginfiles:
        pluginmodule = importlib.import_module(
                'neuralyzer.io.plugins.'+pf.split('.')[0])
        if not hasattr(pluginmodule, 'FILE_EXTENSIONS'):
            continue
        for ext in pluginmodule.FILE_EXTENSIONS:
            ps[ext] = pluginmodule
    return ps


class DataHandler(object):
    '''
    A class taking care of your data (IO).
    '''

    def __init__(self, root_path=''):
        self._root_path = os.path.abspath(root_path)

    @property
    def root_path(self):
        return self._root_path

    def get_data(self, filename):
        extension = os.path.splitext(filename)[1][1:]
        return io_plugins()[extension].Loader.get_data(filename)
