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

* **model_dir** (str): Directory containing trained rate RNN model .mat files
  (default: '../models/go-nogo/P_rec_0.2_Taus_4.0_20.0')
* **n_trials** (int): Number of trials to evaluate each scaling factor
  (default: 100)
* **scaling_factors** (list): List of scaling factors to test
  (default: [20, 25, 30, ..., 75])
* **task_name** (str): Task type ('go-nogo', 'xor', or 'mante')
  (default: 'go-nogo')

Single Trial Evaluation
----------------------------------------------------

The evaluate_single_trial function tests a specific scaling factor:

* **curr_full** (str): Full path to model file
* **scaling_factor** (float): Scaling factor to test
* **trial_params** (dict): Trial parameters including stimulus and task settings
* **task_name** (str): Name of the task to evaluate

Returns performance metrics for the given scaling factor.

Example Usage
----------------------------------------------------

.. code-block:: python

   from spiking import lambda_grid_search

   # Basic grid search with default parameters
   lambda_grid_search()

   # Grid search with custom parameters
   lambda_grid_search(
       model_dir='models/go-nogo',
       n_trials=50,
       scaling_factors=list(range(30, 81, 5)),
       task_name='go-nogo'
   )

   # Evaluate a single trial
   from spiking.lambda_grid_search import evaluate_single_trial
   
   performance = evaluate_single_trial(
       model_path='models/go-nogo/trained_model.mat',
       scaling_factor=50.0,
       trial_params={},
       task_name='go-nogo'
   )

Optimization Process
----------------------------------------------------

The grid search follows these steps:

1. **Load trained rate models** from the specified directory
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

The function modifies each input .mat file to include:

* **opt_scaling_factor**: The optimal scaling factor found
* **all_perfs**: Performance scores for all tested scaling factors
* **scaling_factors**: List of all scaling factors that were tested


Output
----------------------------------------------------

The function outputs:

* Progress updates during optimization
* Performance scores for each scaling factor tested
* Optimal scaling factor and its performance
* Updated model file with optimal scaling factor saved 