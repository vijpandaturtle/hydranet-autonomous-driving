import json
import os
import random
from datetime import datetime
from inspect import signature

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]
        
def load_state_dict(model, state_dict, strict=False):
    if state_dict is None:
        return
    logger = logging.getLogger(__name__)
    # When using dataparallel, 'module.' is prepended to the parameters' keys
    # This function handles the cases when state_dict was saved without DataParallel
    # But the user wants to load it into the model created with dataparallel, and
    # vice versa
    is_module_model_dict = list(model.state_dict().keys())[0].startswith("module")
    is_module_state_dict = list(state_dict.keys())[0].startswith("module")
    if is_module_model_dict and is_module_state_dict:
        pass
    elif is_module_model_dict:
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    elif is_module_state_dict:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    logger.info(model.load_state_dict(state_dict, strict=strict))


class Saver:
    """Saver class for checkpointing the training progress."""

    def __init__(
        self,
        args,
        ckpt_dir,
        best_val=0,
        condition=lambda x, y: x > y,
        save_interval=100,
        save_several_mode=any,
    ):
        """
        Args:
            args (dict): dictionary with arguments.
            ckpt_dir (str): path to directory in which to store the checkpoint.
            best_val (float or list of floats): initial best value.
            condition (function or list of functions): how to decide whether to save
                                                       the new checkpoint by comparing
                                                       best value and new value (x,y).
            save_interval (int): always save when the interval is triggered.
            save_several_mode (any or all): if there are multiple savers, how to trigger
                                            the saving.
        """
        if save_several_mode not in [all, any]:
            raise ValueError(
                f"save_several_mode must be either all or any, got {save_several_mode}"
            )
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        with open("{}/args.json".format(ckpt_dir), "w") as f:
            json.dump(
                {k: self.serialise(v) for k, v in args.items()},
                f,
                sort_keys=True,
                indent=4,
                ensure_ascii=False,
            )
        self.ckpt_dir = ckpt_dir
        self.best_val = make_list(best_val)
        self.condition = make_list(condition)
        self._counter = 0
        self._save_interval = save_interval
        self.save_several_mode = save_several_mode
        self.logger = logging.getLogger(__name__)

    def _do_save(self, new_val):
        """Check whether need to save"""
        do_save = [
            condition(val, best_val)
            for (condition, val, best_val) in zip(
                self.condition, new_val, self.best_val,
            )
        ]
        return self.save_several_mode(do_save)

    def maybe_save(self, new_val, dict_to_save):
        """Maybe save new checkpoint"""
        self._counter += 1
        if "epoch" not in dict_to_save:
            dict_to_save["epoch"] = self._counter
        new_val = make_list(new_val)
        if self._do_save(new_val):
            for (val, best_val) in zip(new_val, self.best_val):
                self.logger.info(
                    " New best value {:.4f}, was {:.4f}".format(val, best_val)
                )
            self.best_val = new_val
            dict_to_save["best_val"] = new_val
            torch.save(dict_to_save, "{}/checkpoint.pth.tar".format(self.ckpt_dir))
            return True
        elif self._counter % self._save_interval == 0:
            self.logger.info(" Saving at epoch {}.".format(dict_to_save["epoch"]))
            dict_to_save["best_val"] = self.best_val
            torch.save(
                dict_to_save, "{}/counter_checkpoint.pth.tar".format(self.ckpt_dir)
            )
            return False
        return False

    def maybe_load(self, ckpt_path, keys_to_load):
        """Loads existing checkpoint if exists.
        Args:
          ckpt_path (str): path to the checkpoint.
          keys_to_load (list of str): keys to load from the checkpoint.
        Returns the epoch at which the checkpoint was saved.
        """
        keys_to_load = make_list(keys_to_load)
        if not os.path.isfile(ckpt_path):
            return [None] * len(keys_to_load)
        ckpt = torch.load(ckpt_path)
        loaded = []
        for key in keys_to_load:
            val = ckpt.get(key, None)
            if key == "best_val" and val is not None:
                self.best_val = make_list(val)
                self.logger.info(f" Found checkpoint with best values {self.best_val}")
            loaded.append(val)
        return loaded

    @staticmethod
    def serialise(x):
        if isinstance(x, (list, tuple)):
            return [Saver.serialise(item) for item in x]
        elif isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, (int, float, str)):
            return x
        elif x is None:
            return x
        else:
            pass
