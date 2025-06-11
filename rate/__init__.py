"""
Rate-based Recurrent Neural Networks for Cognitive Tasks

This module provides continuous-variable rate recurrent neural network (RNN) models
based on the framework from Kim et al. (2019). The rate models can be trained on
various cognitive tasks and subsequently mapped to spiking neural networks.

The module includes:
- FR_RNN_dale: Main firing-rate RNN class with Dale's principle support
- Task generators for Go-NoGo, XOR, and Mante tasks
- Utility functions for GPU management and parameter validation
- Training and evaluation functions
- Abstract base classes for extensibility

Example usage:
```python
import torch
import numpy as np
from rate import FR_RNN_dale, generate_input_stim_go_nogo, generate_target_continuous_go_nogo
from rate import create_default_config, set_gpu

# Set up device
device = set_gpu('0', 0.3)

# Create network configuration
config = create_default_config(N=200, P_inh=0.2, P_rec=0.2, device=str(device))

# Initialize input and output weights
w_in = np.random.randn(200, 1).astype(np.float32)  # Go-NoGo task has 1 input
w_out = np.random.randn(1, 200).astype(np.float32) / 100  # Single output

# Create the rate RNN
net = FR_RNN_dale(config.N, config.P_inh, config.P_rec, w_in, 
                    config.som_N, config.w_dist, config.gain, 
                    config.apply_dale, w_out, device)

# Generate task stimuli
settings = {'T': 200, 'stim_on': 50, 'stim_dur': 25, 'DeltaT': 1}
stim, label = generate_input_stim_go_nogo(settings)
target = generate_target_continuous_go_nogo(settings, label)

# Display network info
net.display()
```

References:
    Kim R., Li Y., & Sejnowski TJ. Simple Framework for Constructing Functional 
    Spiking Recurrent Neural Networks. Proceedings of the National Academy of 
    Sciences. 116: 22811-22820 (2019).
"""

from .model import (
    FR_RNN_dale,
    generate_input_stim_go_nogo,
    generate_input_stim_xor, 
    generate_input_stim_mante,
    generate_target_continuous_go_nogo,
    generate_target_continuous_xor,
    generate_target_continuous_mante,
    loss_op,
    eval_rnn
)

from .utils import (
    set_gpu,
    restricted_float,
    str2bool
)

from .abstract import (
    AbstractRateRNN,
    AbstractTaskGenerator,
    AbstractTargetGenerator,
    AbstractLossFunction,
    RNNConfig,
    RateRNNFactory,
    validate_config,
    create_default_config
)

__version__ = "0.1.0"
__author__ = "NuttidaLab"
__email__ = "nr2869@columbia.edu"

__all__ = [
    # Main RNN class
    "FR_RNN_dale",
    
    # Task stimulus generators
    "generate_input_stim_go_nogo",
    "generate_input_stim_xor", 
    "generate_input_stim_mante",
    
    # Target generators
    "generate_target_continuous_go_nogo",
    "generate_target_continuous_xor",
    "generate_target_continuous_mante",
    
    # Training utilities
    "loss_op",
    "eval_rnn",
    
    # General utilities
    "set_gpu",
    "restricted_float", 
    "str2bool",
    
    # Configuration and factory
    "RNNConfig",
    "RateRNNFactory",
    "validate_config",
    "create_default_config"
] 