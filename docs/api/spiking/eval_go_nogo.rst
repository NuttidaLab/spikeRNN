Go-NoGo Evaluation
==================

.. automodule:: spiking.eval_go_nogo
   :members:
   :undoc-members:
   :show-inheritance:

Main Function
-------------

.. autofunction:: spiking.eval_go_nogo.eval_go_nogo

Description
-----------

The eval_go_nogo module provides functions for evaluating spiking neural network performance 
on the Go-NoGo cognitive task. This task requires the network to respond to "Go" stimuli 
and withhold responses to "NoGo" stimuli, testing impulse control and decision-making capabilities.

The evaluation includes:

* Performance comparison between rate and spiking networks
* Spike raster plot visualization
* Response time analysis
* Accuracy metrics for Go and NoGo trials

Function Parameters
-------------------

The main evaluation function can be called with optional parameters:

* **model_path** (str, optional): Path to trained model file
* **scaling_factor** (float, optional): Scaling factor for conversion
* **n_trials** (int, optional): Number of trials to evaluate
* **plot_results** (bool, optional): Whether to generate plots

Example Usage
-------------

.. code-block:: python

   from spiking import eval_go_nogo

   # Evaluate with default parameters
   eval_go_nogo()

   # Evaluate specific model with custom parameters
   eval_go_nogo(
       model_path='models/go-nogo/model.mat',
       scaling_factor=50.0,
       n_trials=100,
       plot_results=True
   )

Output Metrics
--------------

The evaluation generates the following metrics:

* **Go Trial Accuracy**: Percentage of correct responses to Go stimuli
* **NoGo Trial Accuracy**: Percentage of correct response inhibition to NoGo stimuli
* **Overall Accuracy**: Combined accuracy across all trials
* **Response Time**: Average response time for Go trials
* **Spike Count**: Total number of spikes generated during trials

Visualization
-------------

The function generates several plots:

* Spike raster plots for Go and NoGo trials
* Network output comparison between rate and spiking models
* Performance summary statistics
* Response time distributions 