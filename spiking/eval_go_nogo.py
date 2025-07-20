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
import argparse

from .LIF_network_fnc import LIF_network_fnc
import warnings
warnings.filterwarnings("ignore")


def eval_go_nogo(model_dir='models/go-nogo/P_rec_0.2_Taus_4.0_20.0', optimal_scaling_factor=None):
    # First, load one trained rate RNN
    # Make sure lambda_grid_search.py was performed on the model.
   
    mat_files = [f for f in os.listdir(model_dir) if f.endswith('.mat')]
    model_name = mat_files[0]
    model_path = os.path.join(model_dir, model_name)
    print(f"Using model file: {model_path}")

    # Create a directory for saving plots
    plot_dir = os.path.join(model_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    print(f"Plots will be saved to: {plot_dir}")
    
    # Load model data to get opt_scaling_factor
    model_data = sio.loadmat(model_path)
    if optimal_scaling_factor is None:
        opt_scaling_factor = float(model_data['opt_scaling_factor'].item())
    else:
        opt_scaling_factor = optimal_scaling_factor

    print(f"Optimal scaling factor: {opt_scaling_factor}")
    
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
    nt = params['nt']
        
    t = np.arange(dt, dt*(nt+1), dt)[:nt]

    nogo_out = out.flatten()   # LIF network output
    nogo_rs = rs     # firing rates
    nogo_spk = spk   # spikes

    # --------------------------------------------------------------
    # Go trial example
    # --------------------------------------------------------------
    u = np.zeros((1, 201))
    u[0, 30:50] = 1 
    
    # Run the LIF simulation 
    stims = {'mode': 'none'}
    W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(model_path, scaling_factor,
                                                           u, stims, down_sample, use_initial_weights)
    dt = params['dt']
    T = params['T']
    nt = params['nt']
    
    t = np.arange(dt, dt*(nt+1), dt)[:nt]

    go_out = out.flatten()   # LIF network output
    go_rs = rs     # firing rates
    go_spk = spk   # spikes

    # Load additional model data for plotting
    inh = model_data['inh'].flatten()
    exc = model_data['exc'].flatten()
    N = int(np.array(model_data['N']).squeeze())

    exc_ind = np.where(exc == 1)[0]
    inh_ind = np.where(inh == 1)[0]
    all_ind = np.concatenate((exc_ind, inh_ind))

    # ----------- Output Plot -----------
    plt.figure(figsize=(10, 6))
    plt.plot(t, nogo_out, 'm', linewidth=2, label='NoGo')
    plt.plot(t, go_out, 'g', linewidth=2, label='Go')
    plt.xlabel('Time (s)')
    plt.ylabel('Network Output')
    plt.legend()
    plt.title('Network Output Comparison')
    plt.tight_layout()
    output_filename = os.path.join(plot_dir, "network_output.png")
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved network output plot: {output_filename}")

    # ----------- NoGo spike raster -----------
    plt.figure(figsize=(8, 6))
    for plot_idx, neuron_idx in enumerate(all_ind):
        curr_spk = nogo_spk[neuron_idx, 9:]  # Python 0-based index 9 == 10th timepoint in MATLAB
        spike_indices = np.where(curr_spk > 0)[0]
        if len(spike_indices) > 0:
            spike_times = t[9:][spike_indices]
            color = 'r.' if exc[neuron_idx] == 1 else 'b.'
            plt.plot(spike_times, np.ones(len(spike_times)) * plot_idx, color, markersize=8)
    plt.xlim([0, 1])
    plt.ylim([-5, N+5])
    plt.xlabel('Steps')
    plt.ylabel('Neuron Index')
    plt.title('NoGo Spike Raster (Red: Exc, Blue: Inh)')
    plt.tight_layout()
    nogo_raster_filename = os.path.join(plot_dir, "nogo_spike_raster.png")
    plt.savefig(nogo_raster_filename)
    plt.close()
    print(f"Saved NoGo spike raster: {nogo_raster_filename}")

    # ----------- Go spike raster -----------
    plt.figure(figsize=(8, 6))
    for plot_idx, neuron_idx in enumerate(all_ind):
        curr_spk = go_spk[neuron_idx, 9:]
        spike_indices = np.where(curr_spk > 0)[0]
        if len(spike_indices) > 0:
            spike_times = t[9:][spike_indices]
            color = 'r.' if exc[neuron_idx] == 1 else 'b.'
            plt.plot(spike_times, np.ones(len(spike_times)) * plot_idx, color, markersize=8)
    plt.xlim([0, 1])
    plt.ylim([-5, N+5])
    plt.xlabel('Steps')
    plt.ylabel('Neuron Index')
    plt.title('Go Spike Raster (Red: Exc, Blue: Inh)')
    plt.tight_layout()
    go_raster_filename = os.path.join(plot_dir, "go_spike_raster.png")
    plt.savefig(go_raster_filename)
    plt.close()
    print(f"Saved Go spike raster: {go_raster_filename}")
    print("--- Evaluation complete ---")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Evaluate a trained LIF RNN model constructed to perform the Go-NoGo task.')
    parser.add_argument('--model_dir', type=str, default=_DEFAULT_MODEL_DIR,
                      help='Directory containing the trained rate RNN model .mat files')
    args = parser.parse_args()
    
    eval_go_nogo(model_dir=args.model_dir) 
    
    # Run the following command from the spikeRNN directory:
    """
    python -m spiking.eval_go_nogo \
        --model_dir models/go-nogo/P_rec_0.2_Taus_4.0_20.0
    """
    
    # If you want to use a different scaling factor, run the following command:
    """
    python -m spiking.eval_go_nogo \
        --model_dir models/go-nogo/P_rec_0.2_Taus_4.0_20.0 \
        --optimal_scaling_factor 50.0
    """