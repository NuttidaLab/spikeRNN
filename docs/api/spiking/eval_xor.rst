XOR Evaluation
==============
Functions for evaluating a trained LIF RNN model constructed to perform the XOR task.

.. note::

  This task requires the network to output +1 for "same" inputs and -1 for "different" inputs. It tests the network's ability to perform non-linear computations and integrate sequential information.

  The evaluation includes:

  * Performance analysis across all four XOR input patterns
  * Spike raster plot visualization for each input condition
  * Network output trace comparison
  * Accuracy metrics based on the sign of the network output

Main Function
-------------

.. autofunction:: spiking.eval_xor.eval_xor

Function Parameters
-------------------

The main evaluation script accepts the following command-line or function arguments:

* **model_dir** (str): Path to the directory containing the trained `.mat` model file.
  * (default: 'models/xor/P_rec_0.2_Taus_4.0_20.0')
* **optimal_scaling_factor** (float, optional): A specific scaling factor to use for the spiking conversion.
  * If not provided, the function will load the optimal scaling factor from the model file.

The script will:

1. Load the first `.mat` model file from the specified directory.
2. Load the task settings (`T`, `stim_on`, `delay`, etc.) from the model file.
3. Extract the optimal scaling factor (if not provided).
4. Run simulations for all four XOR input conditions: `(1,1)`, `(1,-1)`, `(-1,1)`, and `(-1,-1)`.
5. Generate and save visualizations of the network's behavior and performance.

Example Usage
-------------

.. code-block:: python

   # In a Python script, assuming you handle the setup
   from spiking import eval_xor, utils
   import os

   # Define model path
   model_dir = 'models/xor/P_rec_0.2_Taus_4.0_20.0'
   model_path = os.path.join(model_dir, os.listdir(model_dir)[0])
   plot_dir = os.path.join(model_dir, 'plots')
   os.makedirs(plot_dir, exist_ok=True)

   # Define task settings
   settings = {
       'T': 301,
       'stim_on': 50,
       'stim_dur': 50,
       'delay': 10
   }
   # Evaluate the model
   eval_xor.eval_xor(
       model_dir=model_dir,
       settings=settings,
       plot_dir=plot_dir
   )

Output Metrics
--------------

The evaluation generates the following metrics:

* **Network Output**: Response curves for all four XOR input patterns.
* **Spike Patterns**: Detailed spike raster plots showing the activity of individual neurons for each condition.
* **Accuracy**: The percentage of trials where the sign of the network's mean output correctly matches the expected label (+1 for "same", -1 for "different").

Visualization
-------------

The function generates and saves several plots to a `plots` subdirectory:

* **Combined Network Outputs**: A single plot comparing the output traces for all four XOR conditions over time.
* **Spike Raster Plots**: A 2x2 grid of plots, with each subplot showing the detailed spiking activity for one of the four input patterns.
    - Excitatory neuron spikes are colored red.
    - Inhibitory neuron spikes are colored blue.
    - Shaded gray regions indicate the timing of stimulus presentation.
