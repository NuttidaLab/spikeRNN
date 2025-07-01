Go-NoGo Evaluation
==================
Functions for evaluating a trained LIF RNN model constructed to perform the Go-NoGo task.

This task requires the network to respond to “Go” stimuli and withhold responses to “NoGo” stimuli, testing impulse control and decision-making capabilities.

The evaluation includes:

* Performance comparison between rate and spiking networks
* Spike raster plot visualization
* Response time analysis
* Accuracy metrics for Go and NoGo trials

Main Function
-------------

.. autofunction:: spiking.eval_go_nogo.eval_go_nogo

Function Parameters
-------------------

The main evaluation function accepts:

* **model_dir** (str): Path to directory containing trained model files
  (default: '../models/go-nogo/P_rec_0.2_Taus_4.0_20.0')

The function will:
1. Load the first .mat model file from the specified directory
2. Extract the optimal scaling factor from the model
3. Run example Go and NoGo trials
4. Generate visualizations of network behavior

Example Usage
-------------

.. code-block:: python

   from spiking import eval_go_nogo

   # Evaluate with default model path
   eval_go_nogo()

   # Evaluate specific model
   eval_go_nogo(
       model_dir='models/go-nogo/my_trained_models'
   )

Output Metrics
--------------

The evaluation generates the following metrics:

* **Network Output**: Response curves for both Go and NoGo trials
* **Spike Patterns**: Detailed spike raster plots showing:

  - Excitatory neuron activity (red)
  - Inhibitory neuron activity (blue)

* **Temporal Dynamics**: Network behavior over the full trial duration

Visualization
-------------

The function generates several plots:

* Network output comparison between Go and NoGo trials
* Spike raster plots showing:

  - NoGo trial neural activity
  - Go trial neural activity
  
* Color-coded neuron types (excitatory in red, inhibitory in blue)
* Time-resolved network responses 