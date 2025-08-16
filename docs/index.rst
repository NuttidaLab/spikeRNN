Welcome to spikeRNN's documentation!
============================================

spikeRNN is a PyTorch framework for constructing functional spiking recurrent neural networks from continuous rate models, based on the framework from Kim et al. (2019) (https://www.pnas.org/doi/10.1073/pnas.1905926116). 

The framework provides two complementary packages with a modern task-based architecture:

* **rate**: Continuous-variable Rate RNN package for training models on cognitive tasks.
* **spiking**: Spiking RNN package for converting rate models to biologically realistic networks.

**Tutorials**

* `spikeRNN Tutorials <tutorials.html>`_ - Framework overview and basic usage
* `How to Create Different Tasks <tasks.html>`_ - Creating and customizing cognitive tasks


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart
   examples


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/rate/index
   api/spiking/index


.. toctree::
   :maxdepth: 1
   :caption: User Guide

   task_architecture
   tutorials
   tasks
   contributing

Indices and tables
--------------------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 