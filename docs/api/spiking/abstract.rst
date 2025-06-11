Spiking Abstract Base Classes
============================================

.. automodule:: spiking.abstract
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
----------------------------------------

.. autoclass:: spiking.abstract.SpikingConfig
   :members:
   :undoc-members:
   :show-inheritance:

Base Classes
----------------------------------------

.. autoclass:: spiking.abstract.AbstractSpikingRNN
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spiking.abstract.AbstractSpikingConverter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spiking.abstract.AbstractSpikingEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

Factory
------------------------

.. autoclass:: spiking.abstract.SpikingRNNFactory
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
----------------------------------------

.. autofunction:: spiking.abstract.validate_spiking_config

.. autofunction:: spiking.abstract.create_default_spiking_config

Description
-----------

The spiking.abstract module provides abstract base classes and interfaces that define a standardized 
framework for implementing spiking neural network models. These abstractions enable extensibility 
and ensure consistent interfaces across different spiking implementations.

The module includes:

* **SpikingConfig**: Dataclass for spiking RNN parameters
* **AbstractSpikingRNN**: Base class for all spiking RNN models
* **AbstractSpikingConverter**: Base class for rate-to-spike converters
* **AbstractSpikingEvaluator**: Base class for spiking network evaluators
* **SpikingRNNFactory**: Factory for creating spiking RNN instances

Key Concepts
------------

**Configuration Management**

The SpikingConfig dataclass standardizes parameter management:

.. code-block:: python

   from spiking.abstract import SpikingConfig, create_default_spiking_config

   # Create default configuration
   config = create_default_spiking_config()

   # Create custom configuration
   config = SpikingConfig(
       N=500,
       scaling_factor=75.0,
       dt=0.00002,
       device='cuda'
   )

**Abstract Base Classes**

All concrete implementations should inherit from the appropriate abstract base class:

.. code-block:: python

   from spiking.abstract import AbstractSpikingRNN

   class MySpikingRNN(AbstractSpikingRNN):
       def __init__(self, config):
           super().__init__(config)
           # Implementation-specific initialization

       def load_rate_weights(self, model_path):
           # Implementation for loading rate weights
           pass

       def simulate(self, stimulus, stims):
           # Implementation for network simulation
           pass

**Factory Pattern**

The factory enables flexible instantiation of different spiking RNN types:

.. code-block:: python

   from spiking.abstract import SpikingRNNFactory

   # Register a new spiking RNN type
   SpikingRNNFactory.register('my_spiking_rnn', MySpikingRNN)

   # Create instance using factory
   config = create_default_spiking_config()
   rnn = SpikingRNNFactory.create('my_spiking_rnn', config)

Configuration Parameters
------------------------

**SpikingConfig Attributes:**

* **N** (int): Number of neurons in the network
* **dt** (float): Integration time step (seconds)
* **downsample** (int): Downsampling factor for simulation
* **scaling_factor** (float): Scaling factor for rate-to-spike conversion
* **tref** (float): Refractory period (seconds)
* **tm** (float): Membrane time constant (seconds)
* **vreset** (float): Reset voltage (mV)
* **vpeak** (float): Spike threshold voltage (mV)
* **tr** (float): Synaptic rise time constant (seconds)
* **use_initial_weights** (bool): Whether to use initial random weights
* **device** (str): PyTorch device for computation

Abstract Method Requirements
----------------------------

**AbstractSpikingRNN Methods:**

* ``load_rate_weights(model_path)``: Load weights from trained rate RNN
* ``initialize_lif_params()``: Initialize LIF neuron parameters
* ``simulate(stimulus, stims)``: Simulate spiking network
* ``compute_firing_rates(spikes)``: Compute firing rates from spikes

**AbstractSpikingConverter Methods:**

* ``convert(rate_model_path, config)``: Convert rate RNN to spiking RNN
* ``optimize_scaling_factor(rate_model_path, task_type, n_trials)``: Optimize scaling

**AbstractSpikingEvaluator Methods:**

* ``evaluate_task_performance(spiking_rnn, task_type, n_trials)``: Evaluate performance
* ``compare_with_rate_model(spiking_rnn, rate_model_path)``: Compare models
* ``analyze_spike_dynamics(spikes, dt)``: Analyze spike dynamics

Example Implementation
----------------------

Here's an example of implementing a custom spiking RNN:

.. code-block:: python

   from spiking.abstract import AbstractSpikingRNN, SpikingConfig
   import torch
   import numpy as np

   class CustomSpikingRNN(AbstractSpikingRNN):
       def __init__(self, config: SpikingConfig):
           super().__init__(config)
           self.weights = None
           self.lif_params = {}

       def load_rate_weights(self, model_path: str):
           model_data = torch.load(model_path)
           self.weights = model_data['model_state_dict']['w']

       def initialize_lif_params(self):
           self.lif_params = {
               'dt': self.config.dt,
               'tref': self.config.tref,
               'tm': self.config.tm,
               'vreset': self.config.vreset,
               'vpeak': self.config.vpeak
           }

       def simulate(self, stimulus, stims):
           # Custom simulation implementation
           # Return spikes, voltages, outputs
           pass

       def compute_firing_rates(self, spikes):
           return np.mean(spikes, axis=1) / self.config.dt

   # Register and use the custom implementation
   from spiking.abstract import SpikingRNNFactory
   SpikingRNNFactory.register('custom', CustomSpikingRNN)
   
   config = create_default_spiking_config(N=200)
   rnn = SpikingRNNFactory.create('custom', config) 