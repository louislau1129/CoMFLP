import torch
import os
import random
import numpy as np

def relocate(x, device):
    ''' move x to the specified device
    '''
    if isinstance(x, list):
        return list([relocate(item, device) for item in x])
    elif isinstance(x, tuple):
        return tuple([relocate(item, device) for item in x])
    elif isinstance(x, dict):
        return dict({k:relocate(v, device) for k,v in x.items()})
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return x.to(device)


def str2bool(s):
    if s.strip().lower() in ["true", "yes"]:
        return True
    else:
        return False


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).
    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import logging
def setup_logger(verbose=1, log_name=None, save_dir=None):
    '''Setup logging config

    Args:
        verbose (int): The value specifiy the logging level, INFO or WARN.
        log_name (str): If specify, the log info will be also written into a 
            log file with this log_name.
        save_dir (str): Only be used if log_name is specified, the log file
            will be stored under this specified directory.
    '''
    log_handlers = [logging.StreamHandler()]
    if log_name is not None:
        # It aslo specifiy to write into a log file.
        if save_dir is None:
            raise RuntimeError('Notice you should specify a save_dir to store the log file.')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        log_file_path = os.path.join(save_dir, log_name)
        if os.path.exists(log_file_path):
            # os.remove(log_file_path)
            pass
        # Set the double handlers when log file is used.
        log_handlers = [logging.FileHandler(log_file_path),
                        logging.StreamHandler()]

    # Set logging.INFO level
    if verbose > 0:
        logging.basicConfig(level=logging.INFO,
                format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                handlers=log_handlers)
        logging.info('Set logging.INFO level.')
    else:
        logging.basicConfig(level=logging.WARN,
                format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                handlers=log_handlers)
        logging.warning('Set logging.WARN level, skip DEBUG/INFO message.')





