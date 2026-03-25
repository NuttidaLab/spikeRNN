Task-Based Architecture
=======================

Overview
--------

spikeRNN introduces a modular task-based architecture that separates cognitive tasks from neural network models. This design makes the framework more extensible, maintainable, and easier to use.

Key Benefits
------------

* **Easy Extensibility**: Add new tasks without modifying core model code
* **Consistent Interface**: All tasks follow the same abstract interface
* **Factory Pattern**: Dynamic task creation and discovery

Architecture Components
-----------------------

Rate Package Tasks
~~~~~~~~~~~~~~~~~~

The ``rate`` package provides the following task classes:

* ``AbstractTask``: Base class for all rate-based tasks
* ``GoNogoTask``: Go/NoGo impulse control task
* ``XORTask``: XOR temporal logic task  
* ``ManteTask``: Context-dependent sensory integration task
* ``TaskFactory``: Factory for creating task instances

Spiking Package Tasks
~~~~~~~~~~~~~~~~~~~~~

The ``spiking`` package provides evaluation tasks:

* ``AbstractSpikingRNN``: Base class for spiking network adapters
* ``GoNogoSpikingEvaluator``: Go/NoGo task evaluation for spiking networks
* ``XORSpikingEvaluator``: XOR task evaluation for spiking networks
* ``ManteSpikingEvaluator``: Mante task evaluation for spiking networks
* ``SpikingEvaluatorFactory``: Factory for creating spiking task evaluator instances
* ``LIFNetworkAdapter``: Adapter for using LIF networks with the evaluation interface

Usage Examples
--------------

Basic Task Usage (Rate)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rate import TaskFactory
    
    # Create task settings
    settings = {
        'T': 200,
        'stim_on': 50,
        'stim_dur': 25,
        'DeltaT': 1
    }
    
    # Create a Go/NoGo task
    task = TaskFactory.create_task('go_nogo', settings)
    
    # Generate stimulus and target
    stimulus, label = task.generate_stimulus()
    target = task.generate_target(label)
    
    # Or generate a complete trial
    stimulus, target, label = task.simulate_trial()

Spiking Task Evaluation
~~~~~~~~~~~~~~~~~~~~~~~

There are two levels of evaluation available:

**Direct task evaluation (when you have a spiking network instance, not necessarily trained)**

.. code-block:: python

    from spiking.eval_tasks import SpikingEvaluatorFactory

    # Create spiking evaluator with task settings
    settings = {'T': 200, 'stim_on': 30, 'stim_dur': 20}
    evaluator = SpikingEvaluatorFactory.create_evaluator('go_nogo', settings)

    # Evaluate a single trial
    result = evaluator.evaluate_single_trial(model_path, scaling_factor)
    print(f"Trial correct: {result}")

**Complete evaluation workflow (when you have a model file (with trained weights))**

.. code-block:: python

    from spiking import evaluate_task
    
    # Complete evaluation including model loading and visualization
    performance = evaluate_task(
        task_name='go_nogo',
        model_path='models/go-nogo/model.mat',
        n_trials=50
    )
    print(f"Performance: {performance}")

**Command-line interface**

.. code-block:: bash

    # Evaluate any task from command line
    python -m spiking.eval_tasks --task go_nogo --model_path models/go-nogo/model.mat
    python -m spiking.eval_tasks --task xor --model_path models/xor/model.mat

Factory Pattern Usage
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
    
    from rate import TaskFactory
    from spiking.eval_tasks import SpikingEvaluatorFactory

    # List available tasks
    print("Rate tasks:", TaskFactory.list_available_tasks())
    print("Spiking evaluators:", SpikingEvaluatorFactory.list_available_tasks())

    # Dynamic task creation
    for task_type in TaskFactory.list_available_tasks():
        task = TaskFactory.create_task(task_type, settings)
        print(f"Created {task_type} task")

Extending the Framework
-----------------------

Adding Custom Tasks
~~~~~~~~~~~~~~~~~~~

To add a new cognitive task, inherit from the appropriate abstract base class:

.. code-block:: python

    from rate.tasks import AbstractTask
    import numpy as np
    
    class MyCustomTask(AbstractTask):
        """Custom cognitive task implementation."""
        
        def validate_settings(self):
            required_keys = ['T', 'custom_param']
            for key in required_keys:
                if key not in self.settings:
                    raise ValueError(f"Missing required setting: {key}")
        
        def generate_stimulus(self, seed=False):
            T = self.settings['T']
            custom_param = self.settings['custom_param']
            
            # Generate custom stimulus
            stimulus = np.random.randn(1, T) * custom_param
            label = "custom_condition"
            return stimulus, label
        
        def generate_target(self, label, seed=False):
            T = self.settings['T']
            # Generate custom target
            target = np.ones(T-1) if label == "custom_condition" else np.zeros(T-1)
            return target

Registering Custom Tasks
~~~~~~~~~~~~~~~~~~~~~~~~

You can extend the factory to include your custom task:

.. code-block:: python

    # Register with factory (optional)
    TaskFactory._registry['my_custom'] = MyCustomTask
    
    # Now you can create it through the factory
    task = TaskFactory.create_task('my_custom', settings)

Task Usage
---------------

Use the task-based API:

.. code-block:: python

    from rate import GoNogoTask
    task = GoNogoTask(settings)
    stimulus, target, label = task.simulate_trial()

Or use the factory:

.. code-block:: python

    from rate import TaskFactory
    task = TaskFactory.create_task('go_nogo', settings)
    stimulus, target, label = task.simulate_trial()

Best Practices
--------------

1. **Use the Factory Pattern**: For maximum flexibility, use ``TaskFactory.create_task()``
2. **Consistent Settings**: Use dictionaries for task settings to ensure consistency
3. **Task Validation**: Always call ``validate_settings()`` in custom task implementations
4. **Error Handling**: Handle ``ValueError`` exceptions from task creation
5. **Documentation**: Document custom task parameters and behavior clearly

Extending Evaluation with Custom Tasks
--------------------------------------

The evaluation system (``eval_tasks.py``) is fully extensible to support custom tasks:

**1. Register Custom Spiking Tasks**

.. code-block:: python

    from spiking.eval_tasks import SpikingEvaluatorFactory
    from rate.tasks import AbstractTask
    
    class MyCustomSpikingEvaluator(AbstractTask):
        def __init__(self, settings):
            super().__init__(settings)
            # Add evaluation-specific settings with defaults
            self.eval_amp_thresh = settings.get('eval_amp_thresh', 0.7)
        
        def validate_settings(self):
            # Validation logic for custom task
            required_keys = ['T', 'custom_param']
            for key in required_keys:
                if key not in self.settings:
                    raise ValueError(f"Missing required setting: {key}")
        
        def evaluate_single_trial(self, model_path: str, scaling_factor: float) -> int:
            """
            Evaluate a single trial for the custom task.
            
            Args:
                model_path: Path to the model .mat file
                scaling_factor: Scaling factor for the model
                
            Returns:
                int: 1 if trial is correct, 0 if incorrect
            """
            # Custom evaluation logic here
            # This should implement the specific task evaluation
            pass
    
    # Register with factory
    SpikingEvaluatorFactory._registry['my_custom'] = MyCustomSpikingEvaluator

**2. Use with eval_tasks.py**

Once registered, your custom task works with the evaluation system:

.. code-block:: bash

    # Command line
    python -m spiking.eval_tasks --task my_custom --model_path models/custom/model.mat
    
.. code-block:: python

    # Programmatic API
    from spiking.eval_tasks import evaluate_task
    
    performance = evaluate_task(
        task_name='my_custom',
        model_path='models/custom/model.mat',
    )

API Reference
-------------

For detailed API documentation, see:

* Rate RNN: :doc:`api/rate/tasks`
* Spiking RNN: :doc:`api/spiking/eval_tasks`

Examples
--------

Complete examples can be found in:

* :doc:`tasks` - Task creation and customization tutorials