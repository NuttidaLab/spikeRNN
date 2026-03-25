import numpy as np
import scipy.io as sio
import os
from multiprocessing import Pool, cpu_count
import argparse
import warnings
import torch
import multiprocessing
from typing import Dict, Any, Optional
multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings("ignore")

def _init_worker():
    """Initialize each worker process with a fresh random seed and clear any GPU state"""

    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    
    # Force CPU computation for worker processes
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    import sys
    import importlib
    custom_modules = ['spiking.LIF_network_fnc', 'spiking.utils']
    for module in custom_modules:
        if module in sys.modules:
            importlib.reload(sys.modules[module])
        
    np.random.seed()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(1)  # Prevent thread contention

def evaluate_single_trial(task_name: str, model_path: str, scaling_factor: float, 
                          settings: Optional[Dict[str, Any]] = None) -> int:
    """
    Evaluate a single trial for a given task using the appropriate evaluator class.
    
    Args:
        task_name: Name of the task ('go_nogo', 'xor', 'mante')
        model_path: Path to the model .mat file
        scaling_factor: Scaling factor for the model
        settings: Optional custom settings. If None, uses default settings.
        
    Returns:
        int: 1 if trial is correct, 0 if incorrect
    """
    from spiking.eval_tasks import SpikingEvaluatorFactory, _get_default_task_settings
    
    try:
        task_name = task_name.replace('-', '_')
        
        # Use provided settings or get default settings for the task
        if settings is None:
            settings = _get_default_task_settings(task_name)
        
        # Create the appropriate evaluator using the factory
        evaluator = SpikingEvaluatorFactory.create_evaluator(task_name, settings)
        
        # Use the evaluator's evaluate_single_trial method
        return evaluator.evaluate_single_trial(model_path, scaling_factor)
        
    except Exception as e:
        print(f"Error in evaluate_single_trial: {e}")
        return 0


def lambda_grid_search(model_path, task_name, n_trials, scaling_factors, 
                       task_settings: Optional[Dict[str, Any]] = None):
    """
    Perform grid search over scaling factors for spiking network evaluation.
    
    Args:
        model_path: Path to the trained model .mat file
        task_name: Name of the task ('go-nogo', 'xor', 'mante')
        n_trials: Number of trials to run for each scaling factor
        scaling_factors: List of scaling factors to test
        task_settings: Optional custom task settings. If None, uses default settings.
                      Can include: T, stim_on, stim_dur, delay, eval_amp_thresh
    
    Returns:
        Optimal scaling factor for the model
    """
    # Verify the file exists and is a .mat file
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not model_path.endswith('.mat'):
        raise ValueError(f"Expected a .mat file, got: {model_path}")
    
    # Create pool with spawn context
    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(initializer=_init_worker)
    
    try:
        # Clear any existing GPU state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        use_initial_weights = False
        down_sample = 1
        mode = 'none'

        if not model_path.endswith('.mat'):
            raise ValueError(f"Expected a .mat file, got: {model_path}")

        model_data = sio.loadmat(model_path)
        # if 'opt_scaling_factor' in model_data:
        #     print("Already processed. Skipping.")
        #     continue
        # else:
        #     model_data['opt_scaling_factor'] = np.nan
        #     sio.savemat(model_path, model_data)
        
        model_data['opt_scaling_factor'] = np.nan
        sio.savemat(model_path, model_data)

        all_perfs = np.zeros(len(scaling_factors))

        # Convert task name format (go-nogo -> go_nogo)
        task_name_normalized = task_name.replace('-', '_')

        for k, scaling_factor in enumerate(scaling_factors):
            print(f"Testing scaling factor: {scaling_factor}")
            
            # Prepare arguments for starmap (each trial gets same arguments)
            trial_args = [(task_name_normalized, model_path, scaling_factor, task_settings) for _ in range(n_trials)]
            
            # Run trials in parallel using multiprocessing
            perfs = pool.starmap(evaluate_single_trial, trial_args)
            all_perfs[k] = np.mean(perfs)
            print(f"Performance for {scaling_factor}: {all_perfs[k]:.3f}")

        best_idx = np.argmax(all_perfs)
        opt_scaling_factor = scaling_factors[best_idx]
        print(f"Best scaling factor: {opt_scaling_factor}")

        model_data = sio.loadmat(model_path)
        model_data['opt_scaling_factor'] = opt_scaling_factor
        model_data['all_perfs'] = all_perfs
        model_data['scaling_factors'] = np.array(scaling_factors)
        sio.savemat(model_path, model_data)
        print("Saved results.")
        return opt_scaling_factor

    except Exception as e:
        print(f"Exception occurred in lambda_grid_search: {e}")
        raise


def parse_range(range_str):
    parts = list(map(int, range_str.split(":")))
    if len(parts) == 3:
        return list(range(parts[0], parts[1] + 1, parts[2]))
    elif len(parts) == 2:
        return list(range(parts[0], parts[1] + 1))
    else:
        raise ValueError("Range must be in 'start:stop:step' or 'start:stop' format")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, choices=["go-nogo", "xor", "mante"], required=True)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--scaling_factors", type=str, default="20:75:5")
    
    # Task-specific settings (advanced usage) - defaults are None so task-specific defaults apply
    parser.add_argument("--T", type=int, help="Trial duration (timesteps)")
    parser.add_argument("--stim_on", type=int, help="Stimulus onset time")
    parser.add_argument("--stim_dur", type=int, help="Stimulus duration")
    parser.add_argument("--delay", type=int, help="Delay between stimuli (XOR task)")
    parser.add_argument("--eval_amp_thresh", type=float, help="Evaluation amplitude threshold")
    
    args = parser.parse_args()

    # Build task settings from arguments
    task_settings = {}
    for param in ['T', 'stim_on', 'stim_dur', 'delay', 'eval_amp_thresh']:
        value = getattr(args, param)
        if value is not None:
            task_settings[param] = value
    
    task_settings = task_settings if task_settings else None

    scaling_factors = parse_range(args.scaling_factors)
    lambda_grid_search(args.model_path, args.task_name, args.n_trials, scaling_factors, task_settings)

    # Run the script with the following command:
    """
    # Basic usage with default settings:
    python -m spiking.lambda_grid_search \
        --model_path "./models/xor/rate_model_xor.mat" \
        --task_name xor \
        --n_trials 50 \
        --scaling_factors 20:76:5
    
    # Advanced usage with custom settings:
    python -m spiking.lambda_grid_search \
        --model_path "./models/go-nogo/rate_model_go_nogo.mat" \
        --task_name go-nogo \
        --n_trials 100 \
        --scaling_factors 20:75:5 \
        --T 200 \
        --stim_on 50 \
        --stim_dur 50 \
        --eval_amp_thresh 0.7
    """