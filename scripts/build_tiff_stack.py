#!/usr/bin/env python
'''
A script to build a .tiff image stack from .tiff image files in a directory.

'''

from __future__ import print_function
import sys

def parse_args(args=sys.argv[1:]):
   import argparse
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument('--filename', default='stack.tif', action='store')
   parser.add_argument('--dir', default='', action='store')
   parser.add_argument('--pattern', default=r'\w*.tif', action='store')
   parser.add_argument('--verbose', action='store_true')
   return parser.parse_args(args)

def main():
    import os
    from neuralyzer.utils import tiffutils
    args = parse_args()
    directory = os.path.abspath(args.dir)
    if args.verbose:
        print("\n\nBuilding tif stack from files in %s matching r'%s'"\
                % (directory, args.pattern))
    imarray, imlist = tiffutils.imarray_from_files_in_directory(
            directory, pat=args.pattern)
    if args.verbose:
        print('\nIncluding the following files:')
        for fn, _ in imlist:
            print(fn)
    filepath = os.path.abspath(args.filename)
    tiffutils.write_tifffile(filepath, imarray)
    if args.verbose:
        print('\nFile written to:  %s' % filepath)

if __name__ == '__main__':
    try:
        main()
    except Exception, err:
        print('\nAn error ocurred:\n')
        print(err)
        print()
