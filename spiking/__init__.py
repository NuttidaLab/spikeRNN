"""
Spiking Neural Networks Package

This module provides leaky integrate-and-fire (LIF) spiking neural network models
mapped from continuous rate RNNs based on the framework from Kim et al. (2019).
The spiking models maintain similar performance to their rate counterparts while
incorporating biologically realistic spike dynamics.

Based on the framework from Kim et al. (2019) for implementing continuous-variable 
rate RNNs that can be mapped to spiking neural networks.

Example:
    Basic rate-to-spike conversion:
    
    >>> import numpy as np
    >>> from spiking import LIF_network_fnc
    >>> 
    >>> # Load trained rate model and convert to spiking
    >>> model_path = 'path/to/trained/model.mat'
    >>> scaling_factor = 50.0
    >>> 
    >>> # Create stimulus
    >>> u = np.zeros((1, 201))
    >>> u[0, 30:50] = 1  # Go trial stimulus
    >>> 
    >>> # Convert and simulate
    >>> W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(
    ...     model_path, scaling_factor, u, {'mode': 'none'}, 
    ...     downsample=1, use_initial_weights=False
    ... )
    >>> 
    >>> print(f"Generated {np.sum(spk)} spikes")

Classes:
    AbstractSpikingRNN: Base class for spiking RNN implementations
    AbstractSpikingConverter: Base class for rate-to-spike converters
    AbstractSpikingEvaluator: Base class for spiking network evaluators
    SpikingConfig: Configuration dataclass for spiking parameters
    SpikingRNNFactory: Factory for creating spiking RNN instances

Functions:
    LIF_network_fnc: Main rate-to-spike conversion function
    lambda_grid_search: Optimize scaling factors via grid search
    evaluate_task: Unified evaluation interface for all tasks
    load_rate_model: Load MATLAB .mat model files
    create_connectivity_masks: Generate network connectivity
    validate_stimulus: Validate input stimulus format
"""

import warnings
from typing import Optional

# Core functions
from .LIF_network_fnc import LIF_network_fnc
from .lambda_grid_search import lambda_grid_search
from .eval_tasks import evaluate_task

# Utility functions 
from .utils import (
    load_rate_model,
    create_connectivity_masks,
    generate_lif_params,
    validate_stimulus,
)

# Abstract base classes and configuration
from .abstract import (
    AbstractSpikingRNN,
    AbstractSpikingConverter, 
    AbstractSpikingEvaluator,
    SpikingConfig,
    SpikingRNNFactory,
    create_default_spiking_config
)

# Task classes
from .tasks import (
    AbstractSpikingTask,
    GoNogoSpikingTask,
    XORSpikingTask, 
    ManteSpikingTask,
    SpikingTaskFactory
)

# Version info
__version__ = "0.1.0"
__author__ = "NuttidaLab"
__email__ = "nr2869@columbia.edu"

# Package metadata
__all__ = [
    # Core spiking functions
    "LIF_network_fnc",
    "lambda_grid_search",
    "evaluate_task",
    
    # Task classes
    "AbstractSpikingTask",
    "GoNogoSpikingTask",
    "XORSpikingTask",
    "ManteSpikingTask", 
    "SpikingTaskFactory",
    
    # Utility functions
    "load_rate_model",
    "create_connectivity_masks",
    "generate_lif_params",
    "validate_stimulus",
    
    # Abstract base classes
    "AbstractSpikingRNN",
    "AbstractSpikingConverter",
    "AbstractSpikingEvaluator",
    
    # Configuration and factory
    "SpikingConfig",
    "SpikingRNNFactory",
    "create_default_spiking_config"
] 