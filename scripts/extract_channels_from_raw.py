#!/usr/bin/env python
'''
A script to extract different channels from raw data files and save them as
tiff stacks.

'''

from __future__ import print_function
import sys, traceback


def parse_args(args=sys.argv[1:]):
   import argparse
   parser = argparse.ArgumentParser(description=__doc__,
           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument('rawfile', action='store',
           help='The raw data file from which the channels are to be extracted.')
   parser.add_argument('--imwidth', default='512', action='store',
           help='The width of the images in the stack.')
   parser.add_argument('--imheight', default='512', action='store',
           help='The height of the images in the stack.')
   parser.add_argument('--verbose', action='store_true',)
   parser.add_argument('--nocache', action='store_true',
           help='Do not generate faster loading cache files.')
   return parser.parse_args(args)


def main():
    import os
    from neuralyzer.utils import rawutils 
    
    args = parse_args()

    # setting up the logger
    from neuralyzer import log
    if args.verbose:
        log.STDOUT_LOGLEVEL = 'DEBUG'
        log.FILE_LOGLEVEL = 'DEBUG'
    else:
        log.STDOUT_LOGLEVEL = 'INFO'
        log.FILE_LOGLEVEL = 'ERROR'
    logger = log.get_logger()

    logger.info("### Extracting channels from raw file %s"\
	    % (args.rawfile))
    logger.debug('args: %s' % args)
    rawutils.extract_channels_from_raw_file(args.rawfile,
            imsize=(int(args.imheight), int(args.imwidth)),
            cache_data=not(args.nocache), logger=logger)


if __name__ == '__main__':
    try:
        main()
    except Exception, err:
	ex_type, ex, tb = sys.exc_info()
        print('\nAn error ocurred: %s\n' % ex)
        traceback.print_tb(tb)
