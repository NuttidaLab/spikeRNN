"""
SpikeRNN: Functional Spiking Recurrent Neural Networks

A comprehensive PyTorch framework for constructing functional spiking recurrent neural
networks from continuous rate models, based on the framework from Kim et al. (2019).

This package provides two main components:
- rate: Continuous-variable rate RNN package for training models on cognitive tasks
- spiking: Spiking RNN package for converting rate models to biologically realistic networks

Example usage:
```python
import spikeRNN

# Train a rate model (or load pre-trained)
from spikeRNN.rate import FR_RNN_dale, create_default_config
config = spikeRNN.rate.create_default_config(N=200, P_inh=0.2)

# Convert to spiking network
from spikeRNN.spiking import LIF_network_fnc
W, REC, spk, rs, all_fr, out, params = spikeRNN.spiking.LIF_network_fnc(
    model_path, scaling_factor, stimulus, stims, downsample, use_initial_weights
)
```

References:
    Kim R., Li Y., & Sejnowski TJ. Simple Framework for Constructing Functional 
    Spiking Recurrent Neural Networks. Proceedings of the National Academy of 
    Sciences. 116: 22811-22820 (2019).
"""

# Version information
__version__ = "0.1.0"
__author__ = "NuttidaLab"
__email__ = "nr2869@columbia.edu"

# Import subpackages
from . import rate
from . import spiking

# Convenience imports from rate package
from .rate import (
    FR_RNN_dale,
    create_default_config as create_rate_config,
    set_gpu
)

# Convenience imports from spiking package
from .spiking import (
    LIF_network_fnc,
    create_default_spiking_config,
    lambda_grid_search,
    eval_go_nogo
)

def check_packages():
    """Check that both packages are available."""
    print("SpikeRNN Package Status:")
    print("=" * 30)
    print("    rate: ✓ Available")
    print(" spiking: ✓ Available")
    print("\nBoth rate and spiking packages are installed and ready to use!")
    return True

# Define what gets exported when someone does "from spikeRNN import *"
__all__ = [
    # Rate RNN essentials
    "FR_RNN_dale",
    "create_rate_config", 
    "set_gpu",
    
    # Spiking RNN essentials
    "LIF_network_fnc",
    "create_default_spiking_config",
    "lambda_grid_search",
    "eval_go_nogo",
    
    # Subpackages
    "rate",
    "spiking",
    
    # Utilities
    "check_packages"
] 