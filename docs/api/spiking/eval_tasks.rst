Evaluation Tasks
================================================================================

.. currentmodule:: spiking.eval_tasks

The eval_tasks module provides a unified, extensible evaluation interface for spiking neural networks on cognitive tasks.

Core Functions
----------------------------------------------------------------------------------

.. autofunction:: evaluate_task

.. autofunction:: load_model_and_scaling_factor

.. autofunction:: main

Adapter Classes
----------------------------------------------------------------------------------

.. autoclass:: LIFNetworkAdapter
   :members:
   :show-inheritance:

Overview
----------------------------------------------------------------------------------

The eval_tasks module provides a high-level evaluation interface that standardizes the process of evaluating trained spiking RNN models across different cognitive tasks. The system is designed to be fully extensible, automatically supporting any task registered with the ``SpikingTaskFactory``.

**Key Features:**

* **Unified Interface**: Single evaluation function for all cognitive tasks
* **Extensible Design**: Automatically supports custom tasks via factory registration
* **Complete Workflow**: Handles model loading, evaluation, and visualization
* **Command-Line Interface**: Convenient CLI for batch evaluation and automation
* **Robust Error Handling**: Graceful handling of evaluation failures
* **Flexible Visualization**: Generic visualization system for any task type

**Evaluation Layers:**

The framework provides three levels of evaluation:

1. **Core Task Methods**: Direct task evaluation (``task.evaluate_performance()``)
2. **High-Level Interface**: Complete workflow (``evaluate_task()``)
3. **Command-Line Interface**: Batch processing (``python -m spiking.eval_tasks``)

Usage Examples
----------------------------------------------------------------------------------

**High-Level Evaluation:**

.. code-block:: python

    from spiking.eval_tasks import evaluate_task
    
    # Evaluate any registered task
    performance = evaluate_task(
        task_name='go_nogo',           # or 'xor', 'mante', custom tasks
        model_dir='models/go-nogo/',
        n_trials=100,
        save_plots=True
    )
    
    print(f"Accuracy: {performance['overall_accuracy']:.3f}")

**Command-Line Interface:**

.. code-block:: bash

    # Basic evaluation
    python -m spiking.eval_tasks --task go_nogo --model_dir models/go-nogo/
    
    # With custom parameters
    python -m spiking.eval_tasks \
        --task xor \
        --model_dir models/xor/ \
        --n_trials 200 \
        --scaling_factor 45.0 \
        --no_plots
    
    # Custom task (after registration)
    python -m spiking.eval_tasks --task my_custom --model_dir models/custom/

**Custom Task Integration:**

.. code-block:: python

    from spiking.tasks import SpikingTaskFactory, AbstractSpikingTask
    from spiking.eval_tasks import evaluate_task
    
    # 1. Define custom task
    class WorkingMemoryTask(AbstractSpikingTask):
        # ... implementation ...
        pass
    
    # 2. Register with factory
    SpikingTaskFactory.register_task('working_memory', WorkingMemoryTask)
    
    # 3. Evaluate using unified interface
    performance = evaluate_task(
        task_name='working_memory',  # Now supported automatically
        model_dir='models/working_memory/',
        n_trials=100
    )

Command-Line Arguments
----------------------------------------------------------------------------------

.. program:: eval_tasks

.. option:: --task TASK

   Task to evaluate. Available tasks are dynamically determined from the factory registry.

.. option:: --model_dir MODEL_DIR

   Directory containing the trained model .mat file.

.. option:: --n_trials N_TRIALS

   Number of trials to evaluate (default: 100).

.. option:: --scaling_factor SCALING_FACTOR

   Override scaling factor (uses value from .mat file if not provided).

.. option:: --no_plots

   Skip generating visualization plots.

.. option:: --T T

   Trial duration (timesteps) - overrides task default.

.. option:: --stim_on STIM_ON

   Stimulus onset time - overrides task default.

.. option:: --stim_dur STIM_DUR

   Stimulus duration - overrides task default.

Implementation Details
----------------------------------------------------------------------------------

**Model Loading:**

The system automatically loads trained rate RNN models from `.mat` files and extracts:

* Network weights and connectivity matrices
* Optimal scaling factors for rate-to-spike conversion
* Task-specific parameters and configurations

**Generic Visualization:**

The visualization system uses each task's ``get_sample_trial_types()`` method to determine what trial types to generate for plotting. This allows custom tasks to specify their own visualization patterns without modifying the evaluation code.

**Error Handling:**

The evaluation system includes comprehensive error handling:

* Graceful handling of missing model files
* Validation of task names against factory registry
* Recovery from trial generation failures
* Informative error messages for debugging

**Extensibility:**

The system is designed to be fully extensible. Any task that inherits from ``AbstractSpikingTask`` and is registered with ``SpikingTaskFactory`` can be evaluated using this unified interface.