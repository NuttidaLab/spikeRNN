Rate RNN Model
============================================

The main FR_RNN_dale class and task-specific functions for creating stimuli and targets.

Main Model Class
----------------------------------------

.. autoclass:: rate.model.FR_RNN_dale
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Task Functions
----------------------------------------

XOR Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rate.model.generate_input_stim_xor
   :noindex:

.. autofunction:: rate.model.generate_target_continuous_xor
   :noindex:

Mante Task  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rate.model.generate_input_stim_mante
   :noindex:

.. autofunction:: rate.model.generate_target_continuous_mante
   :noindex:

Go-NoGo Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rate.model.generate_input_stim_go_nogo
   :noindex:

.. autofunction:: rate.model.generate_target_continuous_go_nogo
   :noindex:

Loss Function
----------------------------------------

.. autofunction:: rate.model.loss_op
   :noindex: 