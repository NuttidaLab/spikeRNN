Rate RNN Model
============================================

The main ``FR_RNN_dale`` class and task-specific functions for creating stimuli and targets.

Main Model Class
----------------------------------------

.. autoclass:: rate.model.FR_RNN_dale 
   :members:
   :undoc-members:
   :show-inheritance:

Core Functions
----------------------------------------

.. autofunction:: rate.model.loss_op

.. autofunction:: rate.model.eval_rnn

Overview
--------

The model module provides the core rate-based RNN implementation with Dale's principle (separate excitatory and inhibitory neurons). 

**Key Components:**

* **FR_RNN_dale**: Main rate RNN class with excitatory/inhibitory neuron types
* **loss_op**: Loss function for training rate RNNs
* **eval_rnn**: Evaluation function for running trained networks

**Note**: Task-specific stimulus and target generation functions have been moved to the :doc:`tasks` module as part of the new task-based architecture. For creating stimuli and targets, use the task classes from ``rate.tasks`` instead. 