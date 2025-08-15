Tasks
================================================================================

.. currentmodule:: rate.tasks

The tasks module provides the task-based architecture for rate RNNs, separating cognitive tasks from neural network models.

Core Classes
----------------------------------------------------------------------------------

.. autoclass:: AbstractTask
   :members:
   :show-inheritance:

.. autoclass:: GoNogoTask
   :members:
   :show-inheritance:

.. autoclass:: XORTask
   :members:
   :show-inheritance:

.. autoclass:: ManteTask
   :members:
   :show-inheritance:

Factory Classes
----------------------------------------------------------------------------------

.. autoclass:: TaskFactory
   :members:
   :show-inheritance:

Overview
----------------------------------------------------------------------------------

The tasks module implements a modern task-based architecture that separates cognitive task logic from neural network implementations. This design promotes modularity, extensibility, and code reuse.

**Key Benefits:**

* **Modular Design**: Tasks are independent of specific neural network implementations
* **Extensible Framework**: Easy to add new cognitive tasks
* **Consistent Interface**: All tasks follow the same ``AbstractTask`` interface
* **Factory Pattern**: Convenient task creation via ``TaskFactory``

**Task Workflow:**

1. **Task Creation**: Use ``TaskFactory.create_task()`` or instantiate directly
2. **Stimulus Generation**: Call ``generate_stimulus()`` to create input patterns
3. **Target Generation**: Call ``generate_target()`` to create expected outputs
4. **Trial Simulation**: Use ``simulate_trial()`` for complete trial workflow

**Supported Tasks:**

* **Go-NoGo**: Binary decision task with response inhibition
* **XOR**: Temporal exclusive OR requiring working memory  
* **Mante**: Context-dependent sensory integration task

Example Usage
----------------------------------------------------------------------------------

.. code-block:: python

    from rate.tasks import TaskFactory
    
    # Create a Go-NoGo task
    settings = {'T': 200, 'stim_on': 50, 'stim_dur': 25}
    task = TaskFactory.create_task('go_nogo', settings)
    
    # Generate specific trial types
    go_stimulus, go_label = task.generate_stimulus('go')
    nogo_stimulus, nogo_label = task.generate_stimulus('nogo')
    
    # Generate targets
    go_target = task.generate_target(go_label)
    nogo_target = task.generate_target(nogo_label)
    
    # Complete trial simulation
    stimulus, target, label = task.simulate_trial()

Custom Task Creation
----------------------------------------------------------------------------------

.. code-block:: python

    from rate.tasks import AbstractTask
    
    class MyCustomTask(AbstractTask):
        def validate_settings(self):
            # Validate required settings
            pass
        
        def generate_stimulus(self, trial_type=None, seed=False):
            # Generate custom stimulus
            return stimulus, label
        
        def generate_target(self, label, seed=False):
            # Generate custom target
            return target
    
    # Register with factory
    TaskFactory.register_task('my_custom', MyCustomTask)