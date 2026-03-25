Scaling Factor Grid Search
====================================================

Functions for optimizing the scaling factor (lambda) used in rate-to-spike conversion. 
The scaling factor is crucial for maintaining task performance when converting from continuous rate dynamics to discrete spiking dynamics.

Main Functions
----------------------------------------------------

.. autofunction:: spiking.lambda_grid_search.lambda_grid_search

.. autofunction:: spiking.lambda_grid_search.evaluate_single_trial

Grid Search Parameters
----------------------------------------------------

The main grid search function accepts:

* ``model_path`` (str): Path to trained rate RNN model .mat file (required)
* ``task_name`` (str): Task type ('go_nogo', 'xor', or 'mante'). Hyphens are accepted and normalized internally (e.g. 'go-nogo' becomes 'go_nogo'). (required)
* ``n_trials`` (int): Number of trials to evaluate each scaling factor (required)
* ``scaling_factors`` (list): List of scaling factors to test (required)
* ``task_settings`` (dict, optional): Custom task settings (T, stim_on, stim_dur, delay, eval_amp_thresh). If None, uses task-specific defaults.

Single Trial Evaluation
----------------------------------------------------

The evaluate_single_trial function tests a specific scaling factor:

* ``task_name`` (str): Name of the task to evaluate ('go_nogo', 'xor', 'mante')
* ``model_path`` (str): Full path to model .mat file
* ``scaling_factor`` (float): Scaling factor to test
* ``settings`` (dict, optional): Custom task settings. If None, uses task-specific defaults.

Returns 1 if the trial is correct, 0 if incorrect.

Example Usage
----------------------------------------------------

.. code-block:: python

   from spiking import lambda_grid_search

   # Grid search with custom parameters
   lambda_grid_search(
       model_path='models/go-nogo/model.mat',
       task_name='go_nogo',
       n_trials=50,
       scaling_factors=list(range(30, 81, 5)),
   )

Optimization Process
----------------------------------------------------

The grid search follows these steps:

1. **Load trained rate model** from the specified .mat file
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

Output Files
----------------------------------------------------

The function modifies the input `.mat` file to include:

* `opt_scaling_factor`: The optimal scaling factor found
* `all_perfs`: Performance scores for all tested scaling factors
* `scaling_factors`: List of all scaling factors that were tested


Output
----------------------------------------------------

The function outputs:

* Progress updates during optimization
* Performance scores for each scaling factor tested
* Optimal scaling factor and its performance
* Updated model file with optimal scaling factor saved 