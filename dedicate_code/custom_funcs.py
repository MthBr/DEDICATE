
"""
custom_funcs.py, we can put in custom code that gets 
used across more than notebook. 
One example would be downstream data preprocessing 
that is only necessary for a subset of notebooks.
"""


from dedicate_code.config import log_dir
from dedicate_code.setup_testSet import setup_dict
file_name=setup_dict['hash_log_name']+'.log'

import logging
import sys
def get_logger(file_name=file_name, logger_name='automated_testing'): #file_path=log_dir/file_name   , display=True
    # create logger
    logger = logging.getLogger(logger_name)

    if len(logger.handlers)<3:
        logger.setLevel(logging.DEBUG)

        # create file handler which logs even debug messages
        fh = logging.FileHandler(log_dir/file_name, mode='a') #a w
        fh.setLevel(logging.DEBUG)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('[%(asctime)s] %(levelname)8s ' +
                                    '- %(name)s - %(relativeCreated)6d - %(threadName)-12s' +
                                    '- %(message)s (%(filename)s:%(lineno)s)',
                                    datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)


        # create file handler which logs even debug messages
        fh2 = logging.FileHandler(log_dir/'general.log', mode='w') #a w
        fh2.setLevel(logging.DEBUG)
        fh2.setFormatter(formatter)
        logger.addHandler(fh2)

        
        # create console handler with a higher log level
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)



    return logger


import time
def benchmark(fn):
    def _timing(*a, **kw):
        st = time.perf_counter()
        r = fn(*a, **kw)
        log = get_logger()
        #print(f"{fn.__name__} execution: {time.perf_counter() - st} seconds")
        log.debug(f"{fn.__name__} execution: {time.perf_counter() - st} seconds")
        return r

    return _timing


def custom_preprocessor(df):  # give the function a more informative name!!!
    """
    Processes the dataframe such that {insert intent here}. (Write better docstrings than this!!!!)

    Intended to be used under this particular circumstance, with {that other function} called before it, and potentially {yet another function} called after it, but optional.

    :param pd.DataFrame df: A pandas dataframe. Should contain the following columns:
        - col1
        - col2
    :returns: A modified dataframe.
    """
    return (df.groupby('col1').count()['col2'])



