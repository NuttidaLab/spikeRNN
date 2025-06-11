LIF Network Function
==================

.. automodule:: spiking.LIF_network_fnc
   :members:
   :undoc-members:
   :show-inheritance:

Function Signature
-------------------

.. py:function:: LIF_network_fnc(model_path, scaling_factor, u, stims, downsample, use_initial_weights)

   Convert a trained rate RNN to a spiking neural network and simulate it.

   :param str model_path: Path to the trained rate RNN model file (.mat format only)
   :param float scaling_factor: Scaling factor for rate-to-spike conversion (typical range: 20-100)
   :param numpy.ndarray u: Input stimulus array with shape (n_inputs, n_timesteps)
   :param dict stims: Stimulation parameters for artificial stimulation
   :param int downsample: Temporal downsampling factor (1 = no downsampling)
   :param bool use_initial_weights: Whether to use initial random weights instead of trained weights
   
   :returns: Tuple containing (W, REC, spk, rs, all_fr, out, params)
   :rtype: tuple

Parameters
------------------------------

* **model_path** (str): Path to the trained rate RNN model file (.mat format)
  
  The model file must contain all necessary parameters for spiking conversion including:
  
  - ``w``: Recurrent weight matrix
  - ``w_in``: Input weight matrix
  - ``w_out``: Output weight matrix
  - ``N``: Number of neurons
  - ``inh``, ``exc``: Inhibitory/excitatory neuron indices
  - ``taus``: Time constants
  - Connectivity masks and other parameters

* **scaling_factor** (float): Controls the conversion intensity from rates to spikes. Higher values produce more spikes but may introduce noise. Typical range is 20-100.

* **u** (numpy.ndarray): Input stimulus with shape (n_inputs, n_timesteps). Each row represents one input channel.

* **stims** (dict): Artificial stimulation parameters with keys:
  
  - ``mode``: Stimulation type ("none", "exc", "inh")
  - ``dur``: Stimulation duration [start_time, end_time] (if applicable)
  - ``units``: List of neuron indices to stimulate (if applicable)

* **downsample** (int): Temporal downsampling factor. Higher values speed up simulation but may reduce accuracy.

* **use_initial_weights** (bool): If True, uses initial random weights instead of trained weights. Mainly for testing purposes.

Returns
----------------------------------------------------

* **W** (numpy.ndarray): Scaled recurrent connectivity matrix (N × N)
* **REC** (numpy.ndarray): Membrane voltage traces for all neurons (timesteps × N)  
* **spk** (numpy.ndarray): Binary spike matrix indicating spike times (N × timesteps)
* **rs** (numpy.ndarray): Instantaneous firing rates for all neurons (N × timesteps)
* **all_fr** (numpy.ndarray): Average firing rates for all neurons (N × 1)
* **out** (numpy.ndarray): Network output signal (1 × timesteps)
* **params** (dict): Simulation parameters including sampling rate and LIF constants

Example Usage
----------------------------------------------------

Basic rate-to-spike conversion:

.. code-block:: python

   import numpy as np
   from spiking import LIF_network_fnc
   
   # Load trained rate model and convert to spiking
   model_path = 'path/to/trained/model.mat'
   scaling_factor = 50.0
   
   # Create Go trial stimulus
   u = np.zeros((1, 201))
   u[0, 30:50] = 1  # 20ms stimulus pulse
   
   # Convert and simulate
   stims = {'mode': 'none'}
   W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(
       model_path, scaling_factor, u, stims,
       downsample=1, use_initial_weights=False
   )
   
   print(f"Generated {np.sum(spk)} spikes")
   print(f"Network output: {out[-1]:.4f}")

With artificial stimulation:

.. code-block:: python

   # Apply excitatory stimulation to specific neurons
   stims = {
       'mode': 'exc',
       'dur': [1000, 1500],  # Stimulate from t=1000 to t=1500
       'units': [10, 15, 20]  # Stimulate neurons 10, 15, 20
   }
   
   W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(
       model_path, scaling_factor, u, stims,
       downsample=1, use_initial_weights=False
   )