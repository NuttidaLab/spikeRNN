Contributing to SpikeRNN
================================================

We welcome contributions to the SpikeRNN framework! This document provides guidelines for contributing code, documentation, and bug reports.

Development Setup
---------------------------------------------------

1. **Fork and Clone**

.. code-block:: bash

   git clone https://github.com/YourUsername/spikeRNN.git
   cd spikeRNN

2. **Set up Development Environment**

.. code-block:: bash

   # Create virtual environment
   python -m venv spikeRNN_dev
   source spikeRNN_dev/bin/activate  # On Windows: spikeRNN_dev\Scripts\activate
   
   # Install in development mode
   cd rate && pip install -e . && cd ..
   cd spiking && pip install -e . && cd ..
   
   # Install development dependencies
   pip install pytest black flake8 mypy sphinx

Package Structure
---------------------------------------------------

The SpikeRNN framework consists of two main packages:

* **rate/**: Rate-based RNN implementation
* **spiking/**: Spiking neural network conversion
* **docs/**: Documentation and examples
* **models/**: Pre-trained model files

Code Style
---------------------------------------------------

We follow PEP 8 style guidelines with some modifications:

**Python Style:**

.. code-block:: python

   # Use descriptive variable names
   network_size = 200
   inhibitory_proportion = 0.2
   
   # Functions should have clear docstrings
   def generate_stimulus(settings: Dict[str, Any]) -> np.ndarray:
       """
       Generate input stimulus for cognitive tasks.
       
       Args:
           settings: Dictionary containing task parameters
           
       Returns:
           Input stimulus array
       """
       pass

**Documentation Style:**

* Use Google-style docstrings
* Include type hints for all function parameters and returns
* Provide examples in docstrings for complex functions
* Keep line length under 88 characters

**Code Organization:**

* Group imports: standard library, third-party, local
* Use meaningful function and variable names
* Add comments for complex logic
* Follow the existing file structure

Testing Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All new functionality should include tests:

.. code-block:: python

   def test_lif_network_conversion():
       """Test basic LIF network conversion functionality."""
       # Test setup
       model_path = 'test_model.mat'
       scaling_factor = 50.0
       
       # Test execution
       result = LIF_network_fnc(model_path, scaling_factor, ...)
       
       # Assertions
       assert result is not None
       assert len(result) == 7  # Expected number of return values

Pull Request Process
---------------------------------------------------

1. **Create Feature Branch**

.. code-block:: bash

   git checkout -b feature/your-feature-name

2. **Make Changes**
   
   * Follow code style guidelines
   * Add tests for new functionality
   * Update documentation as needed

3. **Test Your Changes**

.. code-block:: bash

   # Run tests
   pytest tests/
   
   # Check code style
   black --check .
   flake8 .
   
   # Type checking
   mypy rate/ spiking/

4. **Commit and Push**

.. code-block:: bash

   git add .
   git commit -m "Add feature: descriptive commit message"
   git push origin feature/your-feature-name

5. **Submit Pull Request**
   
   * Provide clear description of changes
   * Link to relevant issues
   * Include screenshots for UI changes

Types of Contributions
---------------------------------------------------

**Bug Fixes:**
* Include reproduction steps
* Add regression tests
* Update documentation if needed

**New Features:**
* Discuss design in an issue first
* Include comprehensive tests
* Update API documentation
* Add examples if appropriate

**Performance Improvements:**
* Include benchmarks showing improvement
* Ensure no functionality regressions
* Document performance characteristics

**Documentation:**
* Fix typos and improve clarity
* Add examples and tutorials
* Update API documentation

Documentation
---------------------------------------------------

Documentation is built using Sphinx. To build locally:

.. code-block:: bash

   cd docs
   make html
   # Open _build/html/index.html in browser

**Documentation Standards:**

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Use clear, concise language
* Include code examples
* Update both docstrings and rst files
* Test all code examples

Bug Reports
---------------------------------------------------

When reporting bugs, please include:

* Python version and OS
* SpikeRNN version
* Minimal reproduction example
* Expected vs actual behavior
* Full error traceback

Feature Requests
---------------------------------------------------

For feature requests:

* Describe the use case
* Explain why existing functionality doesn't meet the need
* Suggest implementation approach
* Consider backward compatibility

Code Review Guidelines
---------------------------------------------------

**For Reviewers:**
* Check code follows style guidelines
* Verify tests cover new functionality
* Ensure documentation is updated
* Test the changes locally

**For Authors:**
* Respond to feedback promptly
* Make requested changes
* Update PR description as needed

Release Process
---------------------------------------------------

1. Update version numbers
2. Update CHANGELOG.md
3. Create release branch
4. Final testing
5. Tag release
6. Publish to PyPI
7. Update documentation

Contact
---------------------------------------------------

* **Issues**: Use GitHub issues for bugs and feature requests
* **Discussions**: Use GitHub discussions for questions
* **Email**: Contact maintainers for security issues

Thank you for contributing to SpikeRNN! 