"""
Functions for evaluating a trained LIF RNN model on the XOR task.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import argparse
from .LIF_network_fnc import LIF_network_fnc
import warnings
warnings.filterwarnings("ignore")

def eval_xor(model_dir='models/xor/', optimal_scaling_factor=None, settings=None):
    """
    Evaluate a trained LIF RNN model on the XOR task.
    """
    mat_files = [f for f in os.listdir(model_dir) if f.endswith('.mat')]
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {model_dir}")
    model_name = mat_files[0]
    model_path = os.path.join(model_dir, model_name)
    print(f"Using model file: {model_path}")

    plot_dir = os.path.join(model_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    print(f"Plots will be saved to: {plot_dir}")

    model_data = sio.loadmat(model_path)
    if optimal_scaling_factor is None:
        if 'opt_scaling_factor' not in model_data:
            raise ValueError("opt_scaling_factor not found in model. Please run lambda_grid_search.py first.")
        opt_scaling_factor = float(model_data['opt_scaling_factor'].item())
    else:
        opt_scaling_factor = optimal_scaling_factor
    print(f"Using scaling factor: {opt_scaling_factor}")

    if settings is None: # default task settings
        T = 400
        stim_on = 50
        stim_dur = 50
        delay = 20
        print(f"Using default task settings: T={T}, stim_on={stim_on}, stim_dur={stim_dur}, delay={delay}")
    else:
        T, stim_on, stim_dur, delay = (
            int(settings['T']), int(settings['stim_on']), int(settings['stim_dur']), int(settings['delay'])
        )
    
    use_initial_weights = False
    down_sample = 1

    # --- Generate Stimulus for one trial type ('++') ---
    u = np.zeros((2, T))
    u[0, stim_on:stim_on+stim_dur] = 1
    u[1, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay] = 1
    
    print("Running simulation for a single '++' trial...")
    stims = {'mode': 'none'}
    W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(
        model_path, opt_scaling_factor, u, stims, down_sample, use_initial_weights
    )
    print("...Simulation finished.")

    # --- Plotting the output ---
    dt = params['dt']
    nt = params['nt']
    t = np.arange(dt, dt*(nt+1), dt)[:nt]
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, out.flatten(), 'g', linewidth=2, label='++ Trial Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Network Output')
    plt.legend()
    plt.title('XOR Network Output')
    plt.tight_layout()
    output_filename = os.path.join(plot_dir, "xor_single_trial_output.png")
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved network output plot: {output_filename}")
    print("--- Evaluation complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained LIF RNN model on the XOR task.')
    parser.add_argument('--model_dir', type=str, default='models/xor/',
                      help='Directory containing the trained model .mat file')
    parser.add_argument('--scaling_factor', type=float, default=None,
                      help='Override scaling factor')
    args = parser.parse_args()
    
    eval_xor(model_dir=args.model_dir, optimal_scaling_factor=args.scaling_factor)
    
    # Example usage:
    """
    python -m spiking.eval_xor \
        --model_dir models/xor/
    """