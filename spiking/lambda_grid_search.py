"""
Functions for optimizing the scaling factor (lambda) used in rate-to-spike conversion. 
The scaling factor is crucial for maintaining task performance 
when converting from continuous rate dynamics to discrete spiking dynamics.

The optimization process:

* Tests multiple scaling factors across a predefined range
* Evaluates spiking network performance for each scaling factor
* Uses parallel processing for efficient computation
* Saves the optimal scaling factor to the model file
* Supports all cognitive tasks (Go-NoGo, XOR, Mante)
"""
# PyTorch adaptation of the script to perform grid search to determine
# the optimal scaling factor (lambda) for one-to-one mapping
# from a trained rate RNN to a LIF RNN

# The original model is from the following paper:
# Kim, R., Hasson, D. V. Z. T., & Pehlevan, C. (2019). A framework for 
# reconciling rate and spike-based neuronal models. arXiv preprint arXiv:1904.05831.
# Original repository: https://github.com/rkim35/spikeRNN

# NOTE
#   - The script utilizes multiprocessing to speed up the script.
#   - Downsampling is turned off (i.e. set to 1). This can be
#   turned on (i.e. setting to a positive integer > 1) to speed up
#   the script, but the resulting LIF network might not be as robust
#   as the one constructed without downsampling.
#   - The script will perform the grid search on all the trained models
#   specified in "model_dir".
#   - For each model in "model_dir", the script computes the task performance
#   for each scaling factor value ("scaling_factors"). The factor value with
#   the best performance is the optimal scaling factor ("opt_scaling_factor").
#   This value is appended to the model mat file.

# Core dependencies
import numpy as np
import scipy.io as sio
import os
from multiprocessing import Pool
import time
from .LIF_network_fnc import LIF_network_fnc

def evaluate_single_trial(args):
    """Helper function for parallel processing of single trials"""
    curr_full, scaling_factor, trial_params, task_name = args
    
    if task_name.lower() == 'go-nogo':
        u = np.zeros((1, 201))
        trial_type = 0
        if np.random.rand() >= 0.50:
            u[0, 50:75] = 1.0  # Python 0-based indexing
            trial_type = 1
        
        stims = {'mode': 'none'}
        down_sample = 1
        use_initial_weights = False
        
        try:
            W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(curr_full, scaling_factor,
                                                                   u, stims, down_sample, use_initial_weights)
            
            perf = 0
            if np.max(out[0, 10000:]) > 0.7 and trial_type == 1:
                perf = 1
            elif np.max(out[0, 10000:]) < 0.3 and trial_type == 0:
                perf = 1
                
            return perf, out
        except:
            return 0, np.zeros((1, 20000))
    
    elif task_name.lower() == 'mante':
        u = np.zeros((4, 501))
        u_lab = np.zeros(2)
        
        # Stim 1
        if np.random.rand() >= 0.50:
            u[0, 50:250] = np.random.randn(200) + 0.5  # Python 0-based indexing
            u_lab[0] = 1
        else:
            u[0, 50:250] = np.random.randn(200) - 0.5
            u_lab[0] = -1
        
        # Stim 2
        if np.random.rand() >= 0.50:
            u[1, 50:250] = np.random.randn(200) + 0.5
            u_lab[1] = 1
        else:
            u[1, 50:250] = np.random.randn(200) - 0.5
            u_lab[1] = -1
        
        # Context
        if np.random.rand() >= 0.50:
            u[2, :] = 1
            label = u_lab[0]
        else:
            u[3, :] = 1
            label = u_lab[1]
        
        stims = {'mode': 'none'}
        down_sample = 1
        use_initial_weights = False
        
        try:
            W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(curr_full, scaling_factor,
                                                                   u, stims, down_sample, use_initial_weights)
            
            perf = 0
            if label == 1:
                if np.max(out[0, 26000:]) > 0.7:
                    perf = 1
            elif label == -1:
                if np.min(out[0, 26000:]) < -0.7:
                    perf = 1
                    
            return perf, out
        except:
            return 0, np.zeros((1, 50000))
    
    elif task_name.lower() == 'xor':
        u = np.zeros((2, 301))
        u_lab = np.zeros(2)
        
        # Stim 1
        if np.random.rand() >= 0.50:
            u[0, 50:100] = 1  # Python 0-based indexing
            u_lab[0] = 1
        else:
            u[0, 50:100] = -1
            u_lab[0] = -1
        
        # Stim 2
        if np.random.rand() >= 0.50:
            u[1, 110:160] = 1
            u_lab[1] = 1
        else:
            u[1, 110:160] = -1
            u_lab[1] = -1
        
        label = np.prod(u_lab)
        
        stims = {'mode': 'none'}
        down_sample = 1
        use_initial_weights = False
        
        try:
            W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(curr_full, scaling_factor,
                                                                   u, stims, down_sample, use_initial_weights)
            
            perf = 0
            if label == 1:
                if np.max(out[0, 20000:]) > 0.7:
                    perf = 1
            elif label == -1:
                if np.min(out[0, 20000:]) < -0.7:
                    perf = 1
                    
            return perf, out
        except:
            return 0, np.zeros((1, 30000))
    
    return 0, np.zeros((1, 1000))

def lambda_grid_search():
    # Directory containing all the trained rate RNN model .mat files
    model_dir = '../models/go-nogo/P_rec_0.2_Taus_4.0_20.0'
    mat_files = [f for f in os.listdir(model_dir) if f.endswith('.mat')]
    
    # Whether to use the initial random connectivity weights
    # This should be set to False unless you want to compare
    # the effects of pre-trained vs post-trained weights
    use_initial_weights = False
    
    # Number of trials to use to evaluate the LIF RNN
    n_trials = 100
    
    # Scaling factor values to try for grid search
    # The more values it has, the longer the search
    scaling_factors = list(range(20, 76, 5))  # [20, 25, 30, ..., 75]
    
    # Grid search
    for i, mat_file in enumerate(mat_files):
        curr_fname = mat_file
        curr_full = os.path.join(model_dir, curr_fname)
        print(f'Analyzing {curr_fname}')
        
        # Get the task name
        task_name = None
        if 'go-nogo' in curr_full:
            task_name = 'go-nogo'
        elif 'mante' in curr_full:
            task_name = 'mante'
        elif 'xor' in curr_full:
            task_name = 'xor'
        
        if task_name is None:
            print(f"Unknown task type for {curr_fname}")
            continue
        
        # Load the model
        model_data = sio.loadmat(curr_full)
        
        # Skip if the file was run before
        if 'opt_scaling_factor' in model_data and not np.isnan(model_data['opt_scaling_factor']).any():
            print(f"Skipping {curr_fname} - already processed")
            continue
        else:
            # Add placeholder for opt_scaling_factor
            model_data['opt_scaling_factor'] = np.nan
            sio.savemat(curr_full, model_data)
        
        print(f"Processing {task_name} task")
        
        all_perfs = np.zeros(len(scaling_factors))
        
        for k, scaling_factor in enumerate(scaling_factors):
            print(f"Testing scaling factor: {scaling_factor}")
            
            # Prepare arguments for parallel processing
            trial_args = [(curr_full, scaling_factor, {}, task_name) for _ in range(n_trials)]
            
            # Use multiprocessing for parallel execution
            # You can adjust the number of processes based on your system
            with Pool(processes=min(8, n_trials)) as pool:
                results = pool.map(evaluate_single_trial, trial_args)
            
            # Extract performances
            perfs = [result[0] for result in results]
            all_perfs[k] = np.mean(perfs)
            
            print(f"Performance for scaling factor {scaling_factor}: {all_perfs[k]:.3f}")
        
        # Find optimal scaling factor
        best_idx = np.argmax(all_perfs)
        best_performance = all_perfs[best_idx]
        opt_scaling_factor = scaling_factors[best_idx]
        
        print(f"Best performance: {best_performance:.3f} with scaling factor: {opt_scaling_factor}")
        
        # Save the optimal scaling factor
        model_data = sio.loadmat(curr_full)
        model_data['opt_scaling_factor'] = opt_scaling_factor
        model_data['all_perfs'] = all_perfs
        model_data['scaling_factors'] = np.array(scaling_factors)
        sio.savemat(curr_full, model_data)
        
        print(f"Saved results for {curr_fname}")

if __name__ == "__main__":
    lambda_grid_search() 