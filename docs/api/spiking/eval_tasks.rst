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

**Evaluation:**

1. **High-Level Interface**: Complete workflow (``evaluate_task()``)
2. **Command-Line Interface**: Batch processing (``python -m spiking.eval_tasks``)

Usage Examples
----------------------------------------------------------------------------------

**High-Level Evaluation:**

.. code-block:: python

    from spiking.eval_tasks import evaluate_task
    
    # Evaluate any registered task
    performance = evaluate_task(
        task_name='go_nogo',
        model_path='models/go-nogo/model.mat',
        n_trials=50
    )
    
    print(f"Performance: {performance}")

**Command-Line Interface:**

.. code-block:: bash

    # Basic evaluation
    python -m spiking.eval_tasks --task go_nogo --model_path models/go-nogo/model.mat
    
    # With custom parameters
    python -m spiking.eval_tasks \
        --task xor \
        --model_path models/xor/model.mat \
        --scaling_factor 45.0 \
        --n_trials 50
    
    # Custom task (after registration)
    python -m spiking.eval_tasks --task my_custom --model_path models/custom/model.mat

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
        task_name='working_memory',
        model_path='models/working_memory/model.mat',
    )

Command-Line Arguments
----------------------------------------------------------------------------------

.. program:: eval_tasks

.. option:: --task TASK

   Task to evaluate. Available tasks are dynamically determined from the factory registry.

.. option:: --model_path MODEL_PATH

   Path to the trained model .mat file.

.. option:: --scaling_factor SCALING_FACTOR

   Override scaling factor (uses value from .mat file if not provided).

.. option:: --n_trials N_TRIALS

   Number of trials to evaluate.

.. option:: --T T

   Trial duration (timesteps) - overrides task default.

.. option:: --delay DELAY

   Delay time (timesteps) - overrides task default.

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