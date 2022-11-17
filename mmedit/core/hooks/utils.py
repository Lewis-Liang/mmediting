import warnings
from copy import deepcopy
from functools import partial

import mmcv
import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class BestRecorderHook(Hook):
    """save and update the iter of best model with highest psnr in  a txt file

    Args:
        runner
        ...
    """
    pass

