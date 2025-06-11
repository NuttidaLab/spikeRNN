Installation
============

SpikeRNN consists of two complementary packages that can be installed independently or together.

Requirements
----------------------------------------------------------------------

**Common Requirements:**

* Python >= 3.7
* PyTorch >= 1.8.0
* NumPy >= 1.16.4
* SciPy >= 1.3.1

**Additional Requirements for Spiking Package:**

* Matplotlib >= 3.0.0

Installing Both Packages
----------------------------------------------------------------------

To install the complete SpikeRNN framework with both rate and spiking packages:

.. code-block:: bash

   git clone https://github.com/NuttidaLab/spikeRNN.git
   cd spikeRNN

   # Install rate RNN package
   cd rate
   pip install -e .
   cd ..

   # Install spiking RNN package  
   cd spiking
   pip install -e .
   cd ..

Installing Individual Packages
----------------------------------------------------------------------

**Rate RNN Package Only:**

.. code-block:: bash

   git clone https://github.com/NuttidaLab/spikeRNN.git
   cd spikeRNN/rate
   pip install -e .

**Spiking RNN Package Only:**

.. code-block:: bash

   git clone https://github.com/NuttidaLab/spikeRNN.git
   cd spikeRNN/spiking
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

Verification
----------------------------------------------------------------------

To verify your installation, you can run:

.. code-block:: python

   import spikeRNN
   spikeRNN.check_packages()

This will show which packages are available:

.. code-block:: text

   SpikeRNN Package Status:
   ==============================
        rate: ✓ Available
     spiking: ✓ Available

GPU Support
----------------------------------------------------------------------

Both packages support GPU acceleration through PyTorch's CUDA integration. No additional installation is required if you have a CUDA-compatible PyTorch installation. 