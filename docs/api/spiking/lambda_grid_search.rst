Scaling Factor Grid Search
====================================================

Functions for optimizing the scaling factor (lambda) used in rate-to-spike conversion. 
The scaling factor is crucial for maintaining task performance 
when converting from continuous rate dynamics to discrete spiking dynamics.

The optimization process:

* Tests multiple scaling factors across a predefined range
* Evaluates spiking network performance for each scaling factor
* Uses parallel processing for efficient computation
* Saves the optimal scaling factor to the model file
* Supports all cognitive tasks (Go-NoGo, XOR, Mante)

Main Functions
----------------------------------------------------

.. autofunction:: spiking.lambda_grid_search.lambda_grid_search

.. autofunction:: spiking.lambda_grid_search.evaluate_single_trial

Grid Search Parameters
----------------------------------------------------

The main grid search function accepts:

* **model_path** (str, optional): Path to trained rate RNN model
* **scaling_range** (tuple, optional): Range of scaling factors to test (default: 20-75)
* **n_trials_per_factor** (int, optional): Number of trials per scaling factor
* **task_type** (str, optional): Task type ('go-nogo', 'xor', 'mante')
* **parallel** (bool, optional): Whether to use parallel processing

Single Trial Evaluation
----------------------------------------------------

The evaluate_single_trial function tests a specific scaling factor:

* **model_path** (str): Path to model file
* **scaling_factor** (float): Scaling factor to test
* **trial_params** (dict): Trial parameters including stimulus and task settings

Returns performance metrics for the given scaling factor.

Example Usage
----------------------------------------------------

.. code-block:: python

   from spiking import lambda_grid_search

   # Basic grid search with default parameters
   lambda_grid_search()

   # Grid search with custom parameters
   lambda_grid_search(
       model_path='models/go-nogo/trained_model.mat',
       scaling_range=(30, 80),
       n_trials_per_factor=50,
       task_type='go-nogo',
       parallel=True
   )

   # Evaluate a single scaling factor
   from spiking.lambda_grid_search import evaluate_single_trial
   
   performance = evaluate_single_trial(
       model_path='models/go-nogo/trained_model.mat',
       scaling_factor=50.0,
       trial_params={'stimulus': stimulus, 'task': 'go-nogo'}
   )

Optimization Process
----------------------------------------------------

The grid search follows these steps:

1. **Load trained rate model** from the specified path
2. **Generate test stimuli** appropriate for the task type
3. **Iterate through scaling factors** in the specified range
4. **Convert to spiking network** for each scaling factor
5. **Evaluate performance** using task-specific metrics
6. **Select optimal scaling factor** based on performance
7. **Save optimal value** to model file for future use

Performance Metrics
----------------------------------------------------

Different metrics are used depending on the task:

**Go-NoGo Task:**
* Response accuracy for Go trials
* Response inhibition accuracy for NoGo trials
* Combined accuracy score

**XOR Task:**
* Output accuracy for all input combinations
* Temporal precision of responses

**Mante Task:**
* Context-dependent decision accuracy
* Sensory integration performance

Parallel Processing
----------------------------------------------------

The module supports parallel processing using Python's multiprocessing:

.. code-block:: python

   # Enable parallel processing (default)
   lambda_grid_search(parallel=True)

   # Disable for debugging
   lambda_grid_search(parallel=False)

Output
----------------------------------------------------

The function outputs:

* Progress updates during optimization
* Performance scores for each scaling factor tested
* Optimal scaling factor and its performance
* Updated model file with optimal scaling factor saved 