Welcome to SpikeRNN's documentation!
============================================

SpikeRNN is a PyTorch framework for constructing functional spiking recurrent neural networks from continuous rate models, based on the framework from [Kim et al. (2019)](https://www.nature.com/articles/s41593-019-0504-8). 

The framework provides two complementary packages:

* **rate**: Continuous-variable Rate RNN package for training models on cognitive tasks
* **spiking**: Spiking RNN package for converting rate models to biologically realistic networks


Installation
-----------------------------------------

**Installing Both Packages:**

.. code-block:: bash

   git clone https://github.com/NuttidaLab/spikeRNN.git
   cd spikeRNN
   
   # Install rate RNN package
   cd rate && pip install -e . && cd ..
   
   # Install spiking RNN package
   cd spiking && pip install -e . && cd ..

**Installing Individual Packages:**

.. code-block:: bash

   # Rate RNN only
   cd rate && pip install -e .
   
   # Spiking RNN only
   cd spiking && pip install -e .

Quick Start
-----------------------------------------

**Complete Workflow Example:**

.. code-block:: python

   import numpy as np
   import torch
   from rate import FR_RNN_dale, create_default_config, set_gpu
   from spiking import LIF_network_fnc, lambda_grid_search

   # Step 1: Train a rate RNN (or load pre-trained model)
   device = set_gpu('0', 0.3)
   config = create_default_config(N=200, P_inh=0.2, P_rec=0.2)

   # Step 2: Load trained model and convert to spiking network
   model_path = 'models/go-nogo/P_rec_0.2_Taus_4.0_20.0/model.mat'

   # Step 3: Run spiking simulation
   scaling_factor = 50.0
   u = np.zeros((1, 201))
   u[0, 30:50] = 1  # Go trial stimulus

   stims = {'mode': 'none'}
   W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(
       model_path, scaling_factor, u, stims, 
       downsample=1, use_initial_weights=False
   )

   print(f"Spiking network output shape: {out.shape}")
   print(f"Total spikes generated: {np.sum(spk)}")

Contents
============================================

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials

API Reference
============================================

.. toctree::
   :maxdepth: 2

   api/rate/index
   api/spiking/index

Indices and tables
============================================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 