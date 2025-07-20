Quick Start Guide
====================================

This guide will get you up and running with the spikeRNN framework quickly using command-line instructions.

Installation
------------------------

First, install the spikeRNN package:

.. code-block:: bash

   git clone https://github.com/NuttidaLab/spikeRNN.git
   cd spikeRNN
   pip install -e .

Basic Workflow
--------------------------------------------------

Step 1: Train a rate RNN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train a rate-based recurrent neural network on the Go-NoGo task. The following command will train a network of 200 neurons and save the trained model as both a `.pth` and a `.mat` file in the `models/go-nogo/` directory.

.. code-block:: bash

   python main.py --gpu 0 --gpu_frac 0.20 \
      --n_trials 5000 --mode train \
      --N 200 --P_inh 0.20 --som_N 0 --apply_dale True \
      --gain 1.5 --task go-nogo --act sigmoid --loss_fn l2 \
      --decay_taus 4 20 --output_dir ../


Step 2: Optimize Scaling Factor and Convert to Spiking Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, find the optimal scaling factor (lambda) to convert the trained rate model into a spiking LIF network. This is a crucial step for maintaining task performance.

Run the following command from the `spikeRNN` directory to perform a grid search for the optimal scaling factor. 
This script will test a range of scaling factors and save the best one to the `.mat` model file.

.. code-block:: bash

   python -m spiking.lambda_grid_search \
        --model_dir models/go-nogo/P_rec_0.2_Taus_4.0_20.0 \
        --task_name go-nogo \
        --n_trials 100 \
        --scaling_factors 20:76:5
        

Step 3: Analyze and Evaluate the Spiking Network
~~~~~~~~~~~~~~~~~~~~~~~

Finally, evaluate the performance of the converted spiking network on the Go-NoGo task. 
This script will use the optimal scaling factor found in the previous step and generate plots comparing the network output and showing spike rasters

.. code-block:: bash

   python -m spiking.eval_go_nogo \
        --model_dir models/go-nogo/P_rec_0.2_Taus_4.0_20.0


Working with Different Tasks
----------------------------

You can train and evaluate the network on different tasks by changing the --task argument.

Go-NoGo Task
~~~~~~~~~~~~

Training: `... --task go-nogo ...`

Evaluation: `python -m spiking.eval_go_nogo --model_dir ...`


XOR Task
~~~~~~~~

Training: `... --task xor ...`

Evaluation: TBD


Mante Task
~~~~~~~~~~

Training: `... --task mante ...`

Evaluation: TBD


Model File Requirements
-----------------------

**Important**: The spiking package only supports MATLAB `.mat` files because they contain complete parameter sets required for accurate spiking conversion:

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
       'taus_gaus0': initia
       
When you run the training command, these files are generated for you, so no manual creation is needed.


Next Steps
----------

- Explore the :doc:`examples` for detailed use cases
- Review the API Reference for all available functions
- Check out advanced features in the individual package documentation:

  - `Rate package <../rate/README.md>`_
  - `Spiking package <../spiking/README.md>`_ 