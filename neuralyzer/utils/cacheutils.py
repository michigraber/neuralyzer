
from ..io import cache, data_handler
from .. import log


def save_cache_for_files_in_directory(directory, pattern=r'\w*.tif$',
        logger=log.get_logger()):

    dh = data_handler.DataHandler(directory)
    pathcont = dh.get_listdir(pat=pattern)

    logger.info('Stepping through %s files now.' % len(pathcont))

    for fn, _ in pathcont:
        try:
            _ = dh.get_data(fn, cache_data=True)
        except Exception, err:
            mess = 'could not load data for file %s : %s' % (fn, err)
            self.logger.error(mess)
