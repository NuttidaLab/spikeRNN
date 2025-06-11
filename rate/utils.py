#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

# Contains several general-purpose utility functions

import os
import torch
import argparse
from typing import Union

def set_gpu(gpu: str, frac: float) -> torch.device:
    """
    Function to specify which GPU to use.

    Args:
        gpu (str): String label for gpu (i.e. '0').
        frac (float): GPU memory fraction (i.e. 0.3 for 30% of the total memory).

    Returns:
        torch.device: PyTorch device object for the specified GPU or CPU.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        # Set memory fraction for PyTorch
        torch.cuda.set_per_process_memory_fraction(frac, device=int(gpu))
        return device
    else:
        return torch.device('cpu')

def restricted_float(x: Union[str, float]) -> float:
    """
    Helper function for restricting input arg to range from 0 to 1.

    Args:
        x (Union[str, float]): String or float representing a number.

    Returns:
        float: The validated float value.

    Raises:
        argparse.ArgumentTypeError: If the value is not in range [0.0, 1.0].
    """
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r no in range [0.0, 1.0]"%(x,))
    return x

def str2bool(v: Union[str, bool]) -> bool:
    """
    Helper function to parse boolean input args.

    Args:
        v (Union[str, bool]): String or boolean representing true or false.

    Returns:
        bool: The parsed boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be parsed as a boolean.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.') 