Spiking RNN
=======================

The spiking package provides leaky integrate-and-fire (LIF) spiking neural networks mapped from continuous rate RNNs.


Core Modules
------------

.. toctree::
   :maxdepth: 1

   lif_network
   tasks
   eval_tasks
   lambda_grid_search
   utils

Module Overview
---------------

**LIF_network_fnc.py**
    Core function for converting rate RNNs to spiking networks and running LIF simulations.

**tasks.py**
    Task-based architecture for spiking neural network evaluation with abstract base classes and concrete task implementations.

**eval_tasks.py**
    Unified, extensible evaluation interface for spiking neural networks on cognitive tasks.

**lambda_grid_search.py**
    Grid search optimization for finding optimal scaling factors in rate-to-spike conversion.

**utils.py**
    Utility functions for model loading, network configuration, and spike data analysis.

Quick Reference
---------------

**Main Functions:**

* ``LIF_network_fnc()``: Core rate-to-spike conversion and simulation
* ``evaluate_task()``: Unified evaluation interface for all tasks
* ``lambda_grid_search()``: Optimize scaling factors

**Task Classes:**

* ``AbstractSpikingTask``: Base class for spiking task evaluation
* ``GoNogoSpikingTask``: Go-NoGo task for spiking networks
* ``XORSpikingTask``: XOR task for spiking networks
* ``ManteSpikingTask``: Mante task for spiking networks
* ``SpikingTaskFactory``: Factory for creating spiking task instances

**Configuration:**

* ``SpikingConfig``: Configuration dataclass for spiking RNN parameters
* ``create_default_spiking_config()``: Create default configuration

**Utility Functions:**

* ``load_rate_model()``: Load rate model from `.mat` file
* ``format_spike_data()``: Format spike data for analysis
* ``SpikingTaskFactory.register_task()``: Register custom tasks
