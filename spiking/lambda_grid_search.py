import numpy as np
import scipy.io as sio
import os
from multiprocessing import Pool, cpu_count
import argparse
import warnings
import torch
import multiprocessing
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

def evaluate_single_trial(args):
    curr_full, scaling_factor, task_name, use_initial_weights, down_sample = args
    
    from spiking.LIF_network_fnc import LIF_network_fnc
    
    model_data = sio.loadmat(curr_full)

    try:
        if task_name == 'go-nogo':
            u = np.zeros((1, 201))
            trial_type = 0
            if np.random.rand() >= 0.50:
                u[0, 50:75] = 1.0
                trial_type = 1
            stims = {'mode': 'none'}
            
            W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(model_data, scaling_factor, u, stims, down_sample, use_initial_weights)   
            
            max_output = np.max(out[10000:])
            
            if trial_type == 1:  # Go trial
                success = max_output > 0.7
            else:  # NoGo trial
                success = max_output < 0.3
            
            return 1 if success else 0

        elif task_name == 'mante':
            u = np.zeros((4, 501))
            u_lab = np.zeros(2)
            if np.random.rand() >= 0.5:
                u[0, 50:250] = np.random.randn(200) + 0.5
                u_lab[0] = 1
            else:
                u[0, 50:250] = np.random.randn(200) - 0.5
                u_lab[0] = -1
            if np.random.rand() >= 0.5:
                u[1, 50:250] = np.random.randn(200) + 0.5
                u_lab[1] = 1
            else:
                u[1, 50:250] = np.random.randn(200) - 0.5
                u_lab[1] = -1
            if np.random.rand() >= 0.5:
                u[2, :] = 1
                label = u_lab[0]
            else:
                u[3, :] = 1
                label = u_lab[1]
            stims = {'mode': 'none'}
            _, _, _, _, _, out, _ = LIF_network_fnc(model_data, scaling_factor, u, stims, down_sample, use_initial_weights)
            if (label == 1 and np.max(out[26000:]) > 0.7) or (label == -1 and np.min(out[26000:]) < -0.7):
                return 1
            return 0

        elif task_name == 'xor':
            u = np.zeros((2, 301))
            u_lab = np.zeros(2)
            if np.random.rand() >= 0.5:
                u[0, 50:100] = 1
                u_lab[0] = 1
            else:
                u[0, 50:100] = -1
                u_lab[0] = -1
            if np.random.rand() >= 0.5:
                u[1, 110:160] = 1
                u_lab[1] = 1
            else:
                u[1, 110:160] = -1
                u_lab[1] = -1
            label = np.prod(u_lab)
            stims = {'mode': 'none'}
            _, _, _, _, _, out, _ = LIF_network_fnc(model_data, scaling_factor, u, stims, down_sample, use_initial_weights)
            
            if (label == 1 and np.max(out[0, 20000:]) > 0.7) or (label == -1 and np.min(out[0, 20000:]) < -0.7):
                return 1
            return 0

    except:
        return 0


def lambda_grid_search(model_dir, task_name, n_trials, scaling_factors):
    # Create pool with spawn context
    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(initializer=_init_worker)
    
    try:
        # Clear any existing GPU state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        use_initial_weights = False
        down_sample = 1
        mat_files = [f for f in os.listdir(model_dir) if f.endswith('.mat')]

        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in directory: {model_dir}")

        for mat_file in mat_files:
            curr_full = os.path.join(model_dir, mat_file)
            print(f"Analyzing {mat_file} for {task_name} task")

            model_data = sio.loadmat(curr_full)
            # if 'opt_scaling_factor' in model_data:
            #     print("Already processed. Skipping.")
            #     continue
            # else:
            #     model_data['opt_scaling_factor'] = np.nan
            #     sio.savemat(curr_full, model_data)
            
            model_data['opt_scaling_factor'] = np.nan
            sio.savemat(curr_full, model_data)

            all_perfs = np.zeros(len(scaling_factors))

            for k, scaling_factor in enumerate(scaling_factors):
                print(f"Testing scaling factor: {scaling_factor}")
                trial_args = [(curr_full, scaling_factor, task_name, use_initial_weights, down_sample) for _ in range(n_trials)]
                
                perfs = pool.map(evaluate_single_trial, trial_args)
                all_perfs[k] = np.mean(perfs)
                print(f"Performance for {scaling_factor}: {all_perfs[k]:.3f}")

            best_idx = np.argmax(all_perfs)
            opt_scaling_factor = scaling_factors[best_idx]
            print(f"Best scaling factor: {opt_scaling_factor}")

            model_data = sio.loadmat(curr_full)
            model_data['opt_scaling_factor'] = opt_scaling_factor
            model_data['all_perfs'] = all_perfs
            model_data['scaling_factors'] = np.array(scaling_factors)
            sio.savemat(curr_full, model_data)
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
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, choices=["go-nogo", "xor", "mante"], required=True)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--scaling_factors", type=str, default="20:75:5")
    args = parser.parse_args()

    scaling_factors = parse_range(args.scaling_factors)
    lambda_grid_search(args.model_dir, args.task_name, args.n_trials, scaling_factors)

    # Run the script with the following command:
    """
    python -m spiking.lambda_grid_search \
        --model_dir "models/go-nogo/P_rec_0.2_Taus_4.0_20.0" \
        --task_name go-nogo \
        --n_trials 100 \
        --scaling_factors 20:76:5
    """