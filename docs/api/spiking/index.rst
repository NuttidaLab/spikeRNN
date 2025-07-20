Spiking RNN
=======================

The spiking package provides leaky integrate-and-fire (LIF) spiking neural networks mapped from continuous rate RNNs.


Core Modules
------------

.. toctree::
   :maxdepth: 1

   lif_network
   eval_go_nogo
   lambda_grid_search
   utils

Module Overview
---------------

**LIF_network_fnc.py**
    Core function for converting rate RNNs to spiking networks and running LIF simulations.

**eval_go_nogo.py**
    Evaluation functions for testing spiking network performance on Go-NoGo tasks.

**lambda_grid_search.py**
    Grid search optimization for finding optimal scaling factors in rate-to-spike conversion.

**utils.py**
    Utility functions for model loading, network configuration, and spike data analysis.

Quick Reference
---------------

**Main Functions:**

* ``LIF_network_fnc()``: Core rate-to-spike conversion and simulation
* ``eval_go_nogo()``: Evaluate Go-NoGo task performance
* ``lambda_grid_search()``: Optimize scaling factors

**Configuration:**

* ``SpikingConfig``: Configuration dataclass for spiking RNN parameters
* ``create_default_spiking_config()``: Create default configuration

**Utility Functions:**

* ``load_rate_model()``: Load rate model from `.mat` file
* ``format_spike_data()``: Format spike data for analysis
