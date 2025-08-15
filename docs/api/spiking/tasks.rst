Tasks
====================

.. currentmodule:: spiking.tasks

The tasks module provides the task-based architecture for spiking RNNs, enabling evaluation of spiking neural networks on cognitive tasks.

Core Classes
------------

.. autoclass:: AbstractSpikingTask
   :members:
   :show-inheritance:

.. autoclass:: GoNogoSpikingTask
   :members:
   :show-inheritance:

.. autoclass:: XORSpikingTask
   :members:
   :show-inheritance:

.. autoclass:: ManteSpikingTask
   :members:
   :show-inheritance:

Factory Classes
---------------

.. autoclass:: SpikingTaskFactory
   :members:
   :show-inheritance:

Overview
--------

The spiking tasks module provides specialized task implementations for evaluating spiking neural networks. These tasks extend the rate-based task framework with spiking-specific evaluation capabilities.

**Key Features:**

* **Spiking-Specific Interface**: Designed for spiking neural network evaluation
* **Performance Metrics**: Multi-trial evaluation with detailed performance analysis
* **Visualization Support**: Built-in plotting and visualization capabilities
* **Extensible Registry**: Dynamic task registration for custom implementations
* **Sample Trial Types**: Configurable trial types for visualization and analysis

**Task Evaluation Workflow:**

1. **Task Creation**: Use ``SpikingTaskFactory.create_task()`` or instantiate directly
2. **Single Trial**: Call ``evaluate_trial()`` for individual trial assessment
3. **Multi-Trial**: Use ``evaluate_performance()`` for comprehensive evaluation
4. **Visualization**: Generate plots with ``create_visualization()``

**Available Tasks:**

* **Go-NoGo**: Impulse control evaluation for spiking networks
* **XOR**: Working memory assessment with temporal logic
* **Mante**: Context-dependent decision making evaluation

Example Usage
-------------

.. code-block:: python

    from spiking.tasks import SpikingTaskFactory
    from spiking.eval_tasks import evaluate_task
    
    # Create a spiking task
    task = SpikingTaskFactory.create_task('go_nogo')
    
    # Generate stimuli for specific trial types
    go_stimulus, go_label = task.generate_stimulus('go')
    nogo_stimulus, nogo_label = task.generate_stimulus('nogo')
    
    # Evaluate with a trained spiking network
    performance = task.evaluate_performance(spiking_rnn, n_trials=100)
    print(f"Accuracy: {performance['overall_accuracy']:.2f}")
    
    # High-level evaluation interface
    performance = evaluate_task(
        task_name='go_nogo',
        model_dir='models/go-nogo/',
        n_trials=100
    )

Custom Spiking Task Creation
---------------------------

.. code-block:: python

    from spiking.tasks import AbstractSpikingTask, SpikingTaskFactory
    
    class MyCustomSpikingTask(AbstractSpikingTask):
        def get_default_settings(self):
            return {'T': 200, 'custom_param': 1.0}
        
        def get_sample_trial_types(self):
            return ['type_a', 'type_b']  # For visualization
        
        def generate_stimulus(self, trial_type=None):
            # Generate stimulus logic
            return stimulus, label
        
        def evaluate_performance(self, spiking_rnn, n_trials=100):
            # Multi-trial performance evaluation
            return {'accuracy': 0.85, 'n_trials': n_trials}
    
    # Register with factory
    SpikingTaskFactory.register_task('my_custom', MyCustomSpikingTask)
    
    # Now works with eval_tasks.py
    # python -m spiking.eval_tasks --task my_custom --model_dir models/custom/

Integration with eval_tasks.py
-----------------------------

The tasks module is fully integrated with the evaluation system:

* **Dynamic Task Discovery**: ``eval_tasks.py`` automatically supports all registered tasks
* **Generic Visualization**: Uses ``get_sample_trial_types()`` for plot generation
* **CLI Support**: Command-line interface adapts to new tasks automatically
* **Error Handling**: Robust error handling for custom task implementations