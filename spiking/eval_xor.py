import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import argparse
from .LIF_network_fnc import LIF_network_fnc
import warnings
warnings.filterwarnings("ignore")

from .utils import validate_stimulus

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MODEL_DIR = os.path.join(_PROJ_ROOT, 'models', 'xor', 'P_rec_0.2_Taus_4.0_20.0')


def eval_xor(model_dir=_DEFAULT_MODEL_DIR, optimal_scaling_factor=None, settings=None):
    """
    Evaluate a trained LIF spiking neural network on the XOR task.

    This function loads a pre-trained rate-based RNN model from a .mat file,
    converts it to a Leaky Integrate-and-Fire (LIF) spiking network using an
    optimal scaling factor, and evaluates its performance on the four
    conditions of the XOR task.

    The XOR task is defined by sequential inputs of +1 or -1, where the
    network should output +1 if the inputs are the same and -1 if they are
    different. The function assesses performance by checking if the sign of
    the network's mean output during the decision period matches the
    expected label.

    Args:
        model_dir (str, optional):
            Path to the directory containing the trained .mat model file.
            Defaults to a pre-defined path within the project structure.
        optimal_scaling_factor (float, optional):
            A specific scaling factor to use for the rate-to-spiking
            conversion. If None, the factor is loaded from the .mat file.
            Defaults to None.

    Returns:
        list[dict]:
            A list of dictionaries, where each dictionary contains the
            detailed results for one of the four XOR input patterns. Keys
            include 'name', 'output_trace', 'spike_raster', 'expected_label',
            'final_output', and 'is_correct'.
    """
    # Load the trained model file
    mat_files = [f for f in os.listdir(model_dir) if f.endswith('.mat')]
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {model_dir}")
    model_name = mat_files[0]
    model_path = os.path.join(model_dir, model_name)
    print(f"Using model file: {model_path}")

    plot_dir = os.path.join(model_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plots will be saved to: {plot_dir}")
    
    model_data = sio.loadmat(model_path)
    
    if settings is None:
        T = 301
        stim_on = 50
        stim_dur = 50
        delay = 10
        print(f"Using default task settings: T={T}, stim_on={stim_on}, stim_dur={stim_dur}, delay={delay}")
    else:
        T, stim_on, stim_dur, delay = (
            settings['T'], settings['stim_on'], settings['stim_dur'], 
            settings['delay']
        )
        print(f"Task Settings: T={T}, stim_on={stim_on}, stim_dur={stim_dur}, delay={delay}")
        
    if optimal_scaling_factor is None:
        if 'opt_scaling_factor' not in model_data:
            raise ValueError("opt_scaling_factor not found in .mat file. Please run lambda_grid_search first.")
        opt_scaling_factor = float(model_data['opt_scaling_factor'].item())
    else:
        opt_scaling_factor = optimal_scaling_factor
    print(f"Using scaling factor: {opt_scaling_factor}")
    
    xor_patterns = [
        ([1, 1], 1, "Input (1, 1) [same]"),
        ([1, -1], -1, "Input (1, -1) [diff]"),
        ([-1, 1], -1, "Input (-1, 1) [diff]"),
        ([-1, -1], 1, "Input (-1, -1) [same]")
    ]
    
    results = []
    
    task_end_time = stim_on + (2 * stim_dur) + delay
    target_onset = task_end_time + 10

    # Loop to test each XOR pattern
    for input_pattern, expected_label, pattern_name in xor_patterns:
        print(f"Testing {pattern_name} -> Expected Label: {expected_label}")
        
        u = np.zeros((2, T))
        u[0, stim_on : stim_on + stim_dur] = input_pattern[0]
        u[1, stim_on + stim_dur + delay : task_end_time] = input_pattern[1]
        
        validate_stimulus(u, task_type='xor')

        stims = {'mode': 'none'}
        _, _, spk, _, _, out, params = LIF_network_fnc(model_path, opt_scaling_factor,
                                                      u, stims, 1, False)
        
        # Ensure the window does not exceed output length
        eval_window_start = min(target_onset, len(out) - 1)
        final_output_mean = np.mean(out[eval_window_start:])
        
        # Determine if the trial was correct based on the sign
        is_correct = (final_output_mean > 0 and expected_label > 0) or \
                     (final_output_mean < 0 and expected_label < 0)
        
        results.append({
            'name': pattern_name,
            'output_trace': out,
            'spike_raster': spk,
            'params': params,
            'expected_label': expected_label,
            'final_output': final_output_mean,
            'is_correct': is_correct
        })
        print(f"  Mean output in decision window: {final_output_mean:.4f} -> Correct: {is_correct}")

    num_correct = sum(r['is_correct'] for r in results)
    accuracy = (num_correct / len(results)) * 100
    
    print(f"\n--- XOR Task Evaluation Summary ---")
    print(f"Accuracy: {num_correct}/{len(results)} ({accuracy:.1f}%)")
    
    # Get parameters for plotting
    N = int(model_data['N'].item())
    inh = model_data['inh'].flatten()
    exc = model_data['exc'].flatten()
    exc_ind = np.where(exc == 1)[0]
    inh_ind = np.where(inh == 1)[0]
    
    # Sort indices for consistent plotting (e.g., all excitatory then all inhibitory)
    plot_neuron_indices = np.concatenate((exc_ind, inh_ind))
    
    # --- PLOTTING ---
    # Combined output comparison
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    nt = results[0]['params']['nt']
    dt = results[0]['params']['dt']
    t = np.arange(nt) * dt
    
    for i, r in enumerate(results):
        plt.plot(t, r['output_trace'], color=colors[i], lw=2, label=f"{r['name']}")
    
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Network Output')
    plt.title(f'XOR Task Outputs (Accuracy: {accuracy:.1f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "xor_combined_outputs.png"), dpi=300)
    plt.close()
    print(f"Saved combined XOR outputs plot.")

    # Spike rasters for each pattern
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    for i, r in enumerate(results):
        ax = axes.flatten()[i]
        spikes = r['spike_raster']
        for plot_idx, neuron_idx in enumerate(plot_neuron_indices):
            spike_times = t[spikes[neuron_idx, :] > 0]
            if len(spike_times) > 0:
                color = 'r' if exc[neuron_idx] else 'b'
                ax.plot(spike_times, np.full_like(spike_times, plot_idx), '|', color=color, ms=4)
        
        ax.axvspan(stim_on*dt*100, (stim_on+stim_dur)*dt*100, color='gray', alpha=0.2)
        ax.axvspan((stim_on+stim_dur+delay)*dt*100, task_end_time*dt*100, color='gray', alpha=0.2)
        ax.set_title(r['name'])
        ax.set_ylim(-1, N)

    fig.supxlabel('Time (s)')
    fig.supylabel('Neuron Index (Sorted by E/I)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('XOR Spike Rasters (Red: Excitatory, Blue: Inhibitory)', fontsize=16)
    plt.savefig(os.path.join(plot_dir, "xor_spike_rasters.png"), dpi=300)
    plt.close()
    print(f"Saved XOR spike rasters.")

    print("--- XOR Evaluation complete ---")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained LIF RNN model for the XOR task.')
    parser.add_argument('--model_dir', type=str, default=_DEFAULT_MODEL_DIR, help='Directory of the trained .mat model file.')
    parser.add_argument('--optimal_scaling_factor', type=float, default=None, help='Manually set the scaling factor.')
    args = parser.parse_args()
    eval_xor(model_dir=args.model_dir, optimal_scaling_factor=args.optimal_scaling_factor)
    
    """
    python -m spiking.eval_xor \
        --model_dir "models/xor/P_rec_0.2_Taus_4.0_20.0" \
        --optimal_scaling_factor 50
    """