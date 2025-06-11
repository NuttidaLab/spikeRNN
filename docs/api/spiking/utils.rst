Spiking Utility Functions
=========================

.. automodule:: spiking.utils
   :members:
   :undoc-members:
   :show-inheritance:

Overview
----------------------------------------------------

The utils module provides essential utility functions for the spiking neural networks package,
including model loading, data validation, and network analysis tools.

Key Features:

* MATLAB .mat model file loading and validation
* Connectivity mask generation for Dale's principle
* Input stimulus validation for different cognitive tasks
* Spike train analysis and formatting
* GPU availability checking
* Random seed management for reproducibility

Model Loading
----------------------------------------------------

.. py:function:: load_rate_model(model_path)

   Load a trained rate RNN model from MATLAB .mat file.
   
   Only .mat files are supported as they contain all necessary parameters for
   accurate rate-to-spike conversion, including connectivity masks, neuron types,
   and time constants.

   :param str model_path: Path to the .mat model file
   :returns: Dictionary containing model parameters
   :rtype: dict
   :raises FileNotFoundError: If model file doesn't exist
   :raises ValueError: If file is not .mat format or corrupted

Example Usage
----------------------------------------------------

Loading and validating models:

.. code-block:: python

   from spiking.utils import load_rate_model, validate_stimulus
   
   # Load .mat model file
   model_data = load_rate_model('model.mat')
   
   # Check required parameters
   required_keys = ['w', 'w_in', 'w_out', 'N', 'inh', 'exc', 'taus']
   missing = [k for k in required_keys if k not in model_data]
   if missing:
       print(f"Warning: Missing parameters {missing}")

Creating connectivity patterns:

.. code-block:: python

   from spiking.utils import create_connectivity_masks
   
   # Generate connectivity for 200-neuron network
   inh, exc, m, som_m = create_connectivity_masks(
       N=200, P_inh=0.2, som_N=10, apply_dale=True, seed=42
   )

Network Configuration
----------------------------------------------------

.. autofunction:: spiking.utils.create_connectivity_masks

.. autofunction:: spiking.utils.generate_lif_params

Validation Functions
----------------------------------------------------

.. autofunction:: spiking.utils.validate_stimulus

.. autofunction:: spiking.utils.validate_scaling_factor

Analysis and Formatting
----------------------------------------------------

.. autofunction:: spiking.utils.format_spike_data

.. autofunction:: spiking.utils.check_gpu_availability

.. autofunction:: spiking.utils.set_random_seed

Description
----------------------------------------------------

The spiking.utils module provides essential utility functions for working with spiking neural networks. 
These functions handle model loading, network configuration, input validation, and spike data analysis.

Key Features
------------

**Model Loading:**
* Support for MATLAB .mat model files
* Automatic validation of model structure and required parameters
* GPU memory management for large models

**Network Configuration:**
* Generate connectivity masks for Dale's principle
* Create default LIF neuron parameters
* Support for somatostatin (SOM) neuron types

**Data Validation:**
* Validate input stimulus format for different tasks
* Check scaling factor ranges and validity
* Ensure compatibility between rate and spiking models

**Spike Analysis:**
* Format spike trains for analysis and visualization
* Calculate firing rates and spike statistics
* Extract spike timing information

Example Usage
----------------------------------------------------

**Loading Models:**

.. code-block:: python

   from spiking.utils import load_rate_model

   # Load .mat model file
   model_data = load_rate_model('model.mat')

**Creating Network Connectivity:**

.. code-block:: python

   from spiking.utils import create_connectivity_masks

   N = 200  # Number of neurons
   P_inh = 0.2  # Proportion of inhibitory neurons
   
   inh, exc, m, som_m = create_connectivity_masks(
       N, P_inh, som_N=10, apply_dale=True, seed=42
   )

**Generating LIF Parameters:**

.. code-block:: python

   from spiking.utils import generate_lif_params

   # Default parameters
   lif_params = generate_lif_params()

   # Custom parameters
   lif_params = generate_lif_params(dt=0.0001, downsample=2)

**Validating Input:**

.. code-block:: python

   from spiking.utils import validate_stimulus, validate_scaling_factor
   import numpy as np

   # Validate Go-NoGo stimulus
   u = np.zeros((1, 201))  # 1 input, 201 timesteps
   validate_stimulus(u, task_type='go-nogo')

   # Validate scaling factor
   validate_scaling_factor(50.0, valid_range=(20.0, 100.0))

**Analyzing Spikes:**

.. code-block:: python

   from spiking.utils import format_spike_data

   # Format spike data for analysis
   spike_data = format_spike_data(spikes, dt=0.00005)

   print(f"Total spikes: {spike_data['total_spikes']}")
   print(f"Mean firing rate: {np.mean(spike_data['firing_rates']):.2f} Hz")

**GPU and Random Seed Management:**

.. code-block:: python

   from spiking.utils import check_gpu_availability, set_random_seed

   # Check GPU availability
   gpu_available, device_name = check_gpu_availability()
   print(f"GPU: {device_name}")

   # Set random seed for reproducibility
   set_random_seed(42)

Function Details
----------------------------------------------------

**load_rate_model(model_path)**
    Load a trained rate RNN model from MATLAB .mat file.

**create_connectivity_masks(N, P_inh, som_N, apply_dale, seed)**
    Generate connectivity masks for inhibitory/excitatory neurons and SOM types.

**generate_lif_params(dt, downsample)**
    Create default LIF neuron parameters with customizable time constants.

**validate_stimulus(u, task_type)**
    Ensure input stimulus matches requirements for specific cognitive tasks.

**format_spike_data(spikes, dt)**
    Convert binary spike matrix to analysis-ready format with timing information.

**check_gpu_availability()**
    Return GPU availability status and device information.

**set_random_seed(seed)**
    Set random seeds for NumPy, PyTorch, and CUDA for reproducible results. 