'''
neuralyzer logging tools.
'''

import sys
import logging

LOGGERNAME = 'neuralyzer'

STDOUT_LOGLEVEL = 'DEBUG'
FILE_LOGLEVEL = 'INFO'

LOGFILE = 'neuralyzer.log'

global_logger = None 

def get_logger():
    '''
    Get the global logger
    '''
    global global_logger
    if global_logger is None:
        logger = _create_logger()
        global_logger = logger
	logger.info('NEURALYZER LOGGER STARTED.')
    else:
        logger = global_logger
    return logger

def _create_logger(**kwargs):
    #logfile = kwargs.get('logfile', LOGFILE)
    #fileloglevel = kwargs.get('fileloglevel', FILE_LOGLEVEL)
    stdoutloglevel = kwargs.get('stdoutloglevel', STDOUT_LOGLEVEL)

    logger = logging.getLogger(LOGGERNAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # FORMATTER
    formatter = logging.Formatter(
            '[ %(asctime)s ] [ %(module)s ] [ %(levelname)s ] : %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

    # LOG to file
#   fh = logging.FileHandler(logfile)
#   fh.setLevel(getattr(logging, fileloglevel))
#   fh.setFormatter(formatter)
#   logger.addHandler(fh)

    # LOG to console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, stdoutloglevel))
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.debug('stdoutloglevel: '+stdoutloglevel)

    return logger
