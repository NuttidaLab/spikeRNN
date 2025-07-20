Installation
============

spikeRNN consists of two complementary packages: :doc:`api/rate/index` and :doc:`api/spiking/index`.

Requirements
----------------------------------------------------------------------

* Python >= 3.7
* PyTorch >= 1.8.0
* NumPy >= 1.16.4
* SciPy >= 1.3.1
* Matplotlib >= 3.0.0

Installing spikeRNN
----------------------------------------------------------------------

To install the complete spikeRNN framework with both rate and spiking packages:

.. code-block:: bash

   git clone https://github.com/NuttidaLab/spikeRNN.git
   cd spikeRNN
   pip install -e .

Development Installation
----------------------------------------------------------------------

For development, you can install additional dependencies for either package:

**Rate RNN Development:**

.. code-block:: bash

   cd rate
   pip install -e ".[dev]"

**Spiking RNN Development:**

.. code-block:: bash

   cd spiking
   pip install -e ".[dev]"

This will install additional packages for testing and development:

* pytest >= 6.0
* pytest-cov
* black
* flake8
* mypy

Documentation Dependencies
----------------------------------------------------------------------

To build the documentation, install the documentation dependencies:

.. code-block:: bash

   pip install -e ".[docs]"

This will install:

* sphinx >= 3.0
* sphinx-book-theme
* sphinxcontrib-napoleon

Parallel Processing Support
----------------------------------------------------------------------

For faster grid search optimization in the spiking package:

.. code-block:: bash

   cd spiking
   pip install -e ".[parallel]"

This will install:

* joblib >= 1.0
* multiprocess >= 0.70

GPU Support
----------------------------------------------------------------------

Both packages support GPU acceleration through PyTorch's CUDA integration. No additional installation is required if you have a CUDA-compatible PyTorch installation.