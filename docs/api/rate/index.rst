Rate RNN
====================

The rate package provides continuous-variable rate RNN implementations for training models on cognitive tasks.

Core Modules
------------

.. toctree::
   :maxdepth: 1

   model
   tasks
   utils

Module Overview
---------------

**model.py**
    Contains the main `FR_RNN_dale` class and backward-compatible task functions.

**tasks.py**
    Task-based architecture with abstract base classes and concrete task implementations for cognitive tasks.

**utils.py**
    Utility functions for GPU management, training helpers, and network configuration.

Quick Reference
---------------

**Main Classes:**

* ``FR_RNN_dale``: Main rate RNN class with Dale's principle
* ``RNNConfig``: Configuration dataclass for rate RNN parameters
* ``AbstractTask``: Base class for all cognitive tasks
* ``TaskFactory``: Factory for creating task instances

**Task Classes:**

* ``GoNogoTask``: Go-NoGo impulse control task
* ``XORTask``: Temporal XOR working memory task
* ``ManteTask``: Context-dependent sensory integration task

**Key Functions:**

* ``set_gpu()``: GPU device configuration
* ``create_default_config()``: Create default configuration
* ``TaskFactory.create_task()``: Create task instances

**Supported Tasks:**

* Go-NoGo: Binary decision task with response inhibition
* XOR: Temporal exclusive OR requiring working memory
* Mante: Context-dependent sensory integration task