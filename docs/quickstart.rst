Quick Start Guide
====================================

This guide will get you up and running with the SpikeRNN framework quickly.

Installation
------------------------

First, install both packages:

.. code-block:: bash

   git clone https://github.com/NuttidaLab/spikeRNN.git
   cd spikeRNN
   pip install -e .

Basic Workflow
--------------------------------------------------

Step 1: Train a Rate RNN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import scipy.io as sio
   from rate import FR_RNN_dale, set_gpu, create_default_config
   from rate.model import generate_input_stim_go_nogo, generate_target_continuous_go_nogo

   # Load trained model
   model_path = 'models/go-nogo/P_rec_0.2_Taus_4.0_20.0/model.mat'
   device = set_gpu('0', 0.3)

   # Set up network configuration
   config = create_default_config(N=200, P_inh=0.2, P_rec=0.2)

   # Create and train network (simplified example)
   net = FR_RNN_dale(200, 0.2, 0.2, w_in, som_N=0, w_dist='gaus',
                     gain=1.5, apply_dale=True, w_out=w_out, device=device)

   # Training loop
   settings = {'T': 200, 'stim_on': 50, 'stim_dur': 25, 'DeltaT': 1, 
              'taus': [10], 'task': 'go-nogo'}

   optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
   for trial in range(1000):
       # Generate stimulus and targets
       u, label = generate_input_stim_go_nogo(settings)
       target = generate_target_continuous_go_nogo(settings, label)
       
       # Forward pass and training
       u_tensor = torch.tensor(u, dtype=torch.float32, device=device)
       outputs = net.forward(u_tensor, settings['taus'], 
                            {'activation': 'sigmoid', 'P_rec': 0.2}, settings)
       
       loss = loss_op(outputs, target, {'activation': 'sigmoid'})
       
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

   # Save model in .mat format for spiking conversion
   model_dict = {
       'w': net.w.detach().cpu().numpy(),
       'w_in': net.w_in.detach().cpu().numpy(),
       'w_out': net.w_out.detach().cpu().numpy(),
       'w0': net.w0.detach().cpu().numpy(),
       'N': 200,
       'm': net.m.cpu().numpy(),
       'som_m': net.som_m.cpu().numpy(),
       'inh': net.inh.cpu().numpy(),
       'exc': net.exc.cpu().numpy(),
       'taus': settings['taus'],
       'taus_gaus': net.taus_gaus.detach().cpu().numpy(),
       'taus_gaus0': net.taus_gaus0.detach().cpu().numpy(),
   }
   sio.savemat('trained_model.mat', model_dict)

Step 2: Convert to Spiking Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from spiking import LIF_network_fnc, lambda_grid_search

   # First, optimize the scaling factor
   lambda_grid_search(
       model_path='trained_model.mat',
       scaling_range=(20, 80),
       n_trials_per_factor=50,
       task_type='go-nogo',
       parallel=True
   )

   # Convert to spiking network with optimal scaling
   scaling_factor = 50.0  # Use value from grid search

   # Create stimulus
   u = np.zeros((1, 201))
   u[0, 30:50] = 1  # Go trial stimulus

   # Convert and simulate
   stims = {'mode': 'none'}
   W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(
       'trained_model.mat', scaling_factor, u, stims,
       downsample=1, use_initial_weights=False
   )

   print(f"Spike conversion completed!")
   print(f"Generated {np.sum(spk)} spikes")
   print(f"Output: {out[-1]:.4f}")

Step 3: Analyze Results
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from spiking import eval_go_nogo, format_spike_data
   import matplotlib.pyplot as plt

   # Evaluate performance
   eval_go_nogo(
       model_path='trained_model.mat',
       scaling_factor=50.0,
       n_trials=100,
       plot_results=True
   )

   # Analyze spike patterns
   spike_data = format_spike_data(spk, params['dt'])
   print(f"Active neurons: {len(spike_data['active_neurons'])}")
   print(f"Mean firing rate: {np.mean(spike_data['firing_rates']):.2f} Hz")

   # Plot spike raster
   plt.figure(figsize=(12, 8))
   spike_times, spike_neurons = np.where(spk)
   plt.scatter(spike_times * params['dt'], spike_neurons, s=1, c='black', alpha=0.6)
   plt.xlabel('Time (s)')
   plt.ylabel('Neuron Index')
   plt.title('Spike Raster Plot')
   plt.show()

Working with Different Tasks
----------------------------

Go-NoGo Task
~~~~~~~~~~~~

.. code-block:: python

   from rate.model import generate_input_stim_go_nogo, generate_target_continuous_go_nogo

   settings = {'T': 200, 'stim_on': 50, 'stim_dur': 25, 'DeltaT': 1, 
              'taus': [10], 'task': 'go-nogo'}

   # Go trial
   u_go = np.zeros((1, 201))
   u_go[0, 30:50] = 1

   # NoGo trial  
   u_nogo = np.zeros((1, 201))
   u_nogo[0, 30:50] = -1

XOR Task
~~~~~~~~

.. code-block:: python

   from rate.model import generate_input_stim_xor, generate_target_continuous_xor

   settings = {'T': 300, 'stim_on': [50, 110], 'stim_dur': 50, 'DeltaT': 1,
              'taus': [10], 'task': 'xor'}

   # XOR stimulus with two sequential inputs
   u = np.zeros((2, 301))
   u[0, 50:100] = 1    # First input
   u[1, 110:160] = -1  # Second input (XOR = 1 × -1 = -1)

Mante Task
~~~~~~~~~~

.. code-block:: python

   from rate.model import generate_input_stim_mante, generate_target_continuous_mante

   settings = {'T': 500, 'stim_on': 50, 'stim_dur': 200, 'DeltaT': 1,
              'taus': [10], 'task': 'mante'}

   # Context-dependent integration
   u = np.zeros((4, 501))
   u[0, 50:250] = np.random.randn(200) + 0.5  # Motion coherence
   u[1, 50:250] = np.random.randn(200) - 0.5  # Color coherence  
   u[2, :] = 1  # Motion context

Model File Requirements
-----------------------

**Important**: The spiking package only supports MATLAB .mat files because they contain complete parameter sets required for accurate spiking conversion:

Required Parameters in .mat Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Complete parameter set for spiking conversion
   model_data = {
       'w': recurrent_weights,          # NxN trained weights
       'w_in': input_weights,           # Nx1 input weights
       'w_out': output_weights,         # 1xN output weights
       'w0': initial_weights,           # NxN initial random weights
       'N': network_size,               # Number of neurons
       'm': connectivity_mask,          # NxN Dale's principle mask
       'som_m': som_mask,              # NxN SOM connectivity mask
       'inh': inhibitory_indices,       # Boolean array for inhibitory neurons
       'exc': excitatory_indices,       # Boolean array for excitatory neurons
       'taus': time_constants,          # Synaptic time constants
       'taus_gaus': gaussian_taus,      # Gaussian time constants
       'taus_gaus0': initial_taus,      # Initial time constants
   }

Saving Models for Spiking Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When training rate models, save them in .mat format:

.. code-block:: python

   import scipy.io as sio

   # After training rate RNN...
   model_dict = {
       'w': net.w.detach().cpu().numpy(),
       'w_in': net.w_in.detach().cpu().numpy(), 
       'w_out': net.w_out.detach().cpu().numpy(),
       'w0': net.w0.detach().cpu().numpy(),
       'N': N,
       'm': net.m.cpu().numpy(),
       'som_m': net.som_m.cpu().numpy(),
       'inh': net.inh.cpu().numpy(),
       'exc': net.exc.cpu().numpy(),
       'taus': settings['taus'],
       'taus_gaus': net.taus_gaus.detach().cpu().numpy(),
       'taus_gaus0': net.taus_gaus0.detach().cpu().numpy(),
   }

   sio.savemat('trained_model.mat', model_dict)

Advanced Usage
--------------

Loading and Validating Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from spiking import load_rate_model

   # Load and validate .mat model
   model_data = load_rate_model('trained_model.mat')

   # Check for required parameters
   required_keys = ['w', 'w_in', 'w_out', 'N', 'inh', 'exc', 'taus']
   missing = [k for k in required_keys if k not in model_data]
   if missing:
       print(f"Warning: Missing critical parameters: {missing}")

Scaling Factor Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from spiking import lambda_grid_search

   # Comprehensive grid search
   lambda_grid_search(
       model_path='models/go-nogo/model.mat',
       scaling_range=(20, 100),     # Wide range
       n_trials_per_factor=100,     # More trials for accuracy
       task_type='go-nogo',
       parallel=True               # Use multiprocessing
   )

Next Steps
----------

- Explore the :doc:`examples` for detailed use cases
- Review the :doc:`api` for all available functions
- Check out advanced features in the individual package documentation:
  - `Rate package <../rate/README.md>`_
  - `Spiking package <../spiking/README.md>`_ 