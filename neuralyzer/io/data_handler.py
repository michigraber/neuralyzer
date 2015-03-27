'''
'''

from __future__ import print_function

import os
import re
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
        if not os.path.isdir(self.root_path):
            raise ValueError('The root_path provided is not a directory.')

    @property
    def root_path(self):
        return self._root_path

    def get_data(self, filename):
        extension = os.path.splitext(filename)[1][1:]
        if os.path.isabs(filename):
            filepath = filename
        else:
            filepath = os.path.join(self.root_path, filename)
        return io_plugins()[extension].Loader.get_data(filepath)

    def get_listdir(self, path='', pat=r''):
        if not os.path.isabs(path):
            path = os.path.join(self.root_path, path)
        if not os.path.isdir(path):
            raise ValueError("The path provided ain't a directory!")
        pathcont = [(fn, os.path.getsize(os.path.join(path, fn)))
                for fn in os.listdir(path) if re.match(pat, fn)]
        return pathcont

    def print_listdir(self, path='', pat=r''):
        pathcont = self.get_listdir(path=path, pat=pat)
        for fn, size in pathcont:
            print('{f} \t {s}'.format(f=fn, s=sizeof_fmt(size)))


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)
