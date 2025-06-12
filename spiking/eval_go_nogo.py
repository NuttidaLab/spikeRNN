"""
Functions for evaluating a trained LIF RNN model constructed to perform the Go-NoGo task.

This task requires the network to respond to “Go” stimuli and withhold responses to “NoGo” stimuli, testing impulse control and decision-making capabilities.

The evaluation includes:

* Performance comparison between rate and spiking networks
* Spike raster plot visualization
* Response time analysis
* Accuracy metrics for Go and NoGo trials
"""

# PyTorch adaptation of the script to evaluate a trained LIF RNN model 
# constructed to perform the Go-NoGo task

# The original model is from the following paper:
# Kim, R., Hasson, D. V. Z. T., & Pehlevan, C. (2019). A framework for 
# reconciling rate and spike-based neuronal models. arXiv preprint arXiv:1904.05831.
# Original  repository: https://github.com/rkim35/spikeRNN


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import time

from .LIF_network_fnc import LIF_network_fnc

def eval_go_nogo():
    # First, load one trained rate RNN
    # Make sure lambda_grid_search.py was performed on the model.
    # Update model_path to point where the trained model is
    model_path = '../models/go-nogo/P_rec_0.2_Taus_4.0_20.0'
    mat_files = [f for f in os.listdir(model_path) if f.endswith('.mat')]
    model_name = mat_files[0]
    model_path = os.path.join(model_path, model_name)
    
    # Load model data to get opt_scaling_factor
    model_data = sio.loadmat(model_path)
    opt_scaling_factor = model_data['opt_scaling_factor'].item()
    
    use_initial_weights = False
    scaling_factor = opt_scaling_factor
    down_sample = 1

    # --------------------------------------------------------------
    # NoGo trial example
    # --------------------------------------------------------------
    u = np.zeros((1, 201))  # input stim

    # Run the LIF simulation 
    stims = {'mode': 'none'}
    W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(model_path, scaling_factor,
                                                           u, stims, down_sample, use_initial_weights)
    dt = params['dt']
    T = params['T']
    t = np.arange(dt, T + dt, dt)

    nogo_out = out   # LIF network output
    nogo_rs = rs     # firing rates
    nogo_spk = spk   # spikes

    # --------------------------------------------------------------
    # Go trial example
    # --------------------------------------------------------------
    u = np.zeros((1, 201))  # input stim
    u[0, 30:50] = 1  # Note: Python uses 0-based indexing, MATLAB uses 1-based

    # Run the LIF simulation 
    stims = {'mode': 'none'}
    W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(model_path, scaling_factor,
                                                           u, stims, down_sample, use_initial_weights)
    dt = params['dt']
    T = params['T']
    t = np.arange(dt, T + dt, dt)

    go_out = out   # LIF network output
    go_rs = rs     # firing rates
    go_spk = spk   # spikes

    # Load additional model data for plotting
    model_data = sio.loadmat(model_path)
    inh = model_data['inh'].flatten()
    exc = model_data['exc'].flatten()
    N = int(model_data['N'].item())

    # --------------------------------------------------------------
    # Plot the network output
    # --------------------------------------------------------------
    plt.figure()
    plt.plot(t, nogo_out.flatten(), 'm', linewidth=2, label='NoGo')
    plt.plot(t, go_out.flatten(), 'g', linewidth=2, label='Go')
    plt.xlabel('Time (s)')
    plt.ylabel('Network Output')
    plt.legend()
    plt.title('Network Output Comparison')
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # Plot the spike raster
    # --------------------------------------------------------------
    # NoGo spike raster
    plt.figure(figsize=(8, 6))
    inh_ind = np.where(inh == 1)[0]
    exc_ind = np.where(exc == 1)[0]
    all_ind = np.arange(N)
    
    for i in range(len(all_ind)):
        curr_spk = nogo_spk[all_ind[i], 10:]  # Skip first 10 time steps
        spike_times = t[10:][curr_spk > 0]  # Get corresponding time points
        if exc[all_ind[i]] == 1:
            plt.plot(spike_times, np.ones(len(spike_times)) * i, 'r.', markersize=8)
        else:
            plt.plot(spike_times, np.ones(len(spike_times)) * i, 'b.', markersize=8)
    
    plt.xlim([0, 1])
    plt.ylim([-5, 205])
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Index')
    plt.title('NoGo Spike Raster (Red: Excitatory, Blue: Inhibitory)')
    plt.tight_layout()
    plt.show()

    # Go spike raster
    plt.figure(figsize=(8, 6))
    for i in range(len(all_ind)):
        curr_spk = go_spk[all_ind[i], 10:]  # Skip first 10 time steps
        spike_times = t[10:][curr_spk > 0]  # Get corresponding time points
        if exc[all_ind[i]] == 1:
            plt.plot(spike_times, np.ones(len(spike_times)) * i, 'r.', markersize=8)
        else:
            plt.plot(spike_times, np.ones(len(spike_times)) * i, 'b.', markersize=8)
    
    plt.xlim([0, 1])
    plt.ylim([-5, 205])
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Index')
    plt.title('Go Spike Raster (Red: Excitatory, Blue: Inhibitory)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    eval_go_nogo() 