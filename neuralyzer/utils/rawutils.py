'''
'''

import os


CHANNEL_FILENAME = '{base}_channel{nr}.tif'

def extract_channels_from_raw_file(filepath, imsize=(512,512), numchannels=4,
        extract_channels=(0,1,), cache_data=True, logger=None):
    ''' Loads a raw image stack file and extracts its stacked channels.
    '''
    import numpy as np
    from ..utils import tiffutils
    from ..io import data_handler, cache
    if logger is None:
        from .. import log
        logger = log.get_logger()

    imstack = data_handler.DataHandler().get_data(filepath, imsize=imsize, cache_data=False)
    logger.debug('raw file %s loaded.' % filepath)
    logger.debug('imstack of size: '+str(imstack.shape))
    channels = np.array((imstack.shape[0]/numchannels)*range(numchannels))
    basefilename, _ = os.path.splitext(filepath)
    logger.debug('extracting channels and saving files now.')
    for channel in extract_channels:
        wherechan = channels == channel
        logger.debug('channel %s : No of frames: %s' % (channel, wherechan.sum()))
        chanstack = imstack[wherechan, :,:]
        fn = CHANNEL_FILENAME.format(base=basefilename, nr=channel)
        tiffutils.write_tifffile(fn, chanstack)
        logger.debug('channel tif stack file saved: '+fn)
        if cache_data:
            dcache = cache.DataCache(fn)
            dcache.save_data_cache(chanstack)
            logger.debug('cache file saved.')
