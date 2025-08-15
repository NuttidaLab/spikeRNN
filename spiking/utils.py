"""
Utility functions for spiking neural networks.

This module provides utility functions for loading rate models,
generating connectivity parameters, and validating inputs for 
spiking neural network simulations.
"""

import torch
import numpy as np
import scipy.io as sio
from typing import Dict, Any, Tuple, Optional, Union
import os
import warnings


def load_rate_model(model_path: str) -> Dict[str, Any]:
    """
    Load a trained rate RNN model from MATLAB .mat file.
    
    Args:
        model_path (str): Path to the .mat model file.
        
    Returns:
        Dict[str, Any]: Dictionary containing model data including:
            - w: Recurrent weight matrix
            - w_in: Input weight matrix  
            - w_out: Output weight matrix
            - N: Number of neurons
            - inh/exc: Inhibitory/excitatory neuron indices
            - taus: Time constants
            - Other training parameters
        
    Raises:
        FileNotFoundError: If model file doesn't exist.
        ValueError: If model file is not a .mat file or is corrupted.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not model_path.endswith('.mat'):
        raise ValueError("Only MATLAB .mat files are supported. "
                        "PyTorch .pth files lack necessary parameters for spiking conversion.")
    
    try:
        model_data = sio.loadmat(model_path)
        
        # Validate required keys for spiking conversion
        required_keys = ['w', 'w_in', 'w_out', 'N', 'inh', 'exc', 'taus']
        missing_keys = [key for key in required_keys if key not in model_data]
        if missing_keys:
            warnings.warn(f"Model file missing some parameters: {missing_keys}. "
                         "This may affect spiking conversion quality.")
        
        return model_data
        
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {str(e)}")
    
def create_connectivity_masks(N: int, P_inh: float = 0.2, som_N: int = 0, 
                            apply_dale: bool = True, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create connectivity masks for spiking network simulation.
    
    Args:
        N (int): Number of neurons.
        P_inh (float): Proportion of inhibitory neurons.
        som_N (int): Number of somatostatin neurons.
        apply_dale (bool): Whether to apply Dale's principle.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - inh: Boolean array for inhibitory neurons
            - exc: Boolean array for excitatory neurons  
            - m: Sign mask matrix for Dale's principle
            - som_m: SOM connectivity mask
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Assign excitatory/inhibitory neurons
    if apply_dale:
        inh = np.random.rand(N, 1) < P_inh
        exc = ~inh
    else:
        inh = np.random.rand(N, 1) < 0  # No inhibitory neurons
        exc = ~inh
    
    # Create Dale's principle mask
    m = np.ones((N, N), dtype=np.float32)
    if apply_dale:
        inh_indices = np.where(inh.flatten())[0]
        m[inh_indices, :] = -1
    
    # Create SOM mask
    som_m = np.ones((N, N), dtype=np.float32)
    if som_N > 0 and apply_dale:
        som_inh_indices = np.where(inh.flatten())[0][:som_N]
        inh_indices = np.where(inh.flatten())[0]
        for i in som_inh_indices:
            som_m[i, inh_indices] = 0
    
    return inh.flatten(), exc.flatten(), m, som_m


def generate_lif_params(dt: float = 0.00005, downsample: int = 1) -> Dict[str, float]:
    """
    Generate default LIF neuron parameters.
    
    Args:
        dt (float): Integration time step.
        downsample (int): Downsampling factor.
        
    Returns:
        Dict[str, float]: Dictionary of LIF parameters.
    """
    return {
        'dt': dt * downsample,
        'tref': 0.002,      # Refractory period (s)
        'tm': 0.010,        # Membrane time constant (s)
        'vreset': -65,      # Reset voltage (mV)
        'vpeak': -40,       # Spike threshold (mV)
        'tr': 0.002         # Synaptic rise time (s)
    }


def validate_stimulus(u: np.ndarray, task_type: str = 'go-nogo') -> bool:
    """
    Validate input stimulus format for different tasks.
    
    Args:
        u (np.ndarray): Input stimulus array.
        task_type (str): Type of task ('go-nogo', 'xor', 'mante').
        
    Returns:
        bool: True if stimulus is valid.
        
    Raises:
        ValueError: If stimulus format is invalid.
    """
    if not isinstance(u, np.ndarray):
        raise ValueError("Stimulus must be a numpy array")
    
    if len(u.shape) != 2:
        raise ValueError("Stimulus must be a 2D array (n_inputs, n_timesteps)")
    
    task_requirements = {
        'go-nogo': (1, None),  # 1 input, any length
        'xor': (2, None),      # 2 inputs, any length  
        'mante': (4, None)     # 4 inputs, any length
    }
    
    if task_type.lower() in task_requirements:
        required_inputs, required_length = task_requirements[task_type.lower()]
        
        if u.shape[0] != required_inputs:
            raise ValueError(f"Task '{task_type}' requires {required_inputs} input(s), got {u.shape[0]}")
        
        if required_length is not None and u.shape[1] != required_length:
            raise ValueError(f"Task '{task_type}' requires {required_length} timesteps, got {u.shape[1]}")
    
    return True


def check_gpu_availability() -> Tuple[bool, str]:
    """
    Check GPU availability and return device information.
    
    Returns:
        Tuple[bool, str]: (is_available, device_name)
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return True, device_name
    else:
        return False, "CPU"


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def format_spike_data(spikes: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    """
    Format spike data for analysis and visualization.
    
    Args:
        spikes (np.ndarray): Binary spike matrix (N_neurons, N_timesteps).
        dt (float): Time step size.
        
    Returns:
        Dict[str, np.ndarray]: Dictionary containing formatted spike data.
    """
    N, T = spikes.shape
    
    # Find spike times and neuron indices
    spike_times = []
    spike_neurons = []
    
    for i in range(N):
        spike_indices = np.where(spikes[i, :] > 0)[0]
        spike_times.extend(spike_indices * dt)
        spike_neurons.extend([i] * len(spike_indices))
    
    return {
        'spike_times': np.array(spike_times),
        'spike_neurons': np.array(spike_neurons),
        'firing_rates': np.sum(spikes, axis=1) / (T * dt),
        'total_spikes': np.sum(spikes)
    }


def validate_scaling_factor(scaling_factor: float, valid_range: Tuple[float, float] = (1.0, 1000.0)) -> bool:
    """
    Validate scaling factor for rate-to-spike conversion.
    
    Args:
        scaling_factor (float): Scaling factor value.
        valid_range (Tuple[float, float]): Valid range for scaling factor.
        
    Returns:
        bool: True if scaling factor is valid.
        
    Raises:
        ValueError: If scaling factor is invalid.
    """
    if not isinstance(scaling_factor, (int, float)):
        raise ValueError("Scaling factor must be a number")
    
    if scaling_factor <= 0:
        raise ValueError("Scaling factor must be positive")
    
    min_val, max_val = valid_range
    if scaling_factor < min_val or scaling_factor > max_val:
        warnings.warn(f"Scaling factor {scaling_factor} is outside recommended range {valid_range}")
    
    return True 