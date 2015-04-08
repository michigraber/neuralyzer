'''
'''

from .. import log


CHANNEL_FILENAME = '{base}_channel{nr}.tiff'

def extract_channels_from_raw_file(filepath, imsize=(512,512), numchannels=4,
        extract_channels=(0,1), cache_data=True, logger=log.get_logger()):
    ''' Loads a raw image stack file and extracts its stacked channels.
    '''
    from ..io import data_handler, cache
    imstack = data_handler.DataHandler().get_data(filepath, imsize=imsize)
    channels = np.array(imstack.shape[0]/numchannels)*range(numchannels))
    basefilename, _ = os.path.splitext(filepath)
    for channel in extract_channels:
        chanstack = imstack[channel == channel, :,:]
        fn = CHANNEL_FILENAME.format(base=basefilename, nr=channel)
        tiffutils.write_tifffile(fn, chanstack)
        if cache_data:
            dcache = cache.DataCache(fn)
            dcache.save_data_cache(chanstack)
