Rate RNN Package API
====================

The rate package provides continuous-variable rate RNN implementations for training models on cognitive tasks.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   model
   utils
   abstract

Module Overview
---------------

**model.py**
    Contains the main FR_RNN_dale class and task-specific functions for creating stimuli and targets.

**utils.py**
    Utility functions for GPU management, training helpers, and network configuration.

**abstract.py**
    Abstract base classes and configuration management for rate RNNs.

Quick Reference
---------------

**Main Classes:**

* ``FR_RNN_dale``: Main rate RNN class with Dale's principle
* ``RNNConfig``: Configuration dataclass for rate RNN parameters
* ``AbstractRateRNN``: Base class for rate RNN implementations

**Key Functions:**

* ``set_gpu()``: GPU device configuration
* ``create_default_config()``: Create default configuration
* ``generate_input_stim_**()``: Task stimulus generation functions
* ``generate_target_continuous_*()``: Task target generation functions

**Supported Tasks:**

* Go-NoGo: Binary decision task with response inhibition
* XOR: Temporal exclusive OR requiring working memory
* Mante: Context-dependent sensory integration task 