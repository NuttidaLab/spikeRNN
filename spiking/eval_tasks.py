#!/usr/bin/env python3
"""
Evaluation script for trained spiking RNN models on cognitive tasks.
"""

import argparse
import os
import sys
import scipy.io as sio
import numpy as np
from typing import Dict, Any, Optional

from .LIF_network_fnc import LIF_network_fnc
from .tasks import SpikingTaskFactory
from .abstract import AbstractSpikingRNN


class LIFNetworkAdapter(AbstractSpikingRNN):
    """
    Adapter to use LIF_network_fnc with the spiking task interface.
    """
    
    def __init__(self, model_path: str, scaling_factor: float):
        # Create a minimal config for the abstract class
        from .abstract import SpikingConfig
        config = SpikingConfig(N=200)  # Default N, will be overridden by actual model
        super().__init__(config)
        
        self.model_path = model_path
        self.scaling_factor = scaling_factor
        self.use_initial_weights = False
        self.downsample = 1
    
    def load_rate_weights(self, model_path: str) -> None:
        """Load weights from a trained rate RNN model."""
        # This is handled by LIF_network_fnc internally
        pass
    
    def initialize_lif_params(self) -> None:
        """Initialize LIF neuron parameters."""
        # This is handled by LIF_network_fnc internally
        pass
    
    def compute_firing_rates(self, spikes: np.ndarray) -> np.ndarray:
        """Compute firing rates from spike trains."""
        # Simple firing rate computation
        if spikes.size == 0:
            return np.array([])
        return np.mean(spikes, axis=0) if spikes.ndim > 1 else np.mean(spikes)
    
    def simulate(self, stimulus: np.ndarray, stims: Dict[str, Any]):
        """Simulate the LIF network on given stimulus."""
        W, REC, spikes, rs, all_fr, output, params = LIF_network_fnc(
            self.model_path, 
            self.scaling_factor, 
            stimulus, 
            stims, 
            self.downsample, 
            self.use_initial_weights
        )
        return spikes, None, output, params


def load_model_and_scaling_factor(model_dir: str, optimal_scaling_factor: Optional[float] = None) -> tuple:
    """
    Load model file and determine scaling factor.
    
    Args:
        model_dir: Directory containing the .mat model file
        optimal_scaling_factor: Override scaling factor if provided
        
    Returns:
        Tuple of (model_path, scaling_factor)
    """
    # Find .mat file
    mat_files = [f for f in os.listdir(model_dir) if f.endswith('.mat')]
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {model_dir}")
    
    model_path = os.path.join(model_dir, mat_files[0])
    print(f"Using model file: {model_path}")
    
    # Load scaling factor
    if optimal_scaling_factor is not None:
        scaling_factor = optimal_scaling_factor
        print(f"Using provided scaling factor: {scaling_factor}")
    else:
        model_data = sio.loadmat(model_path)
        if 'opt_scaling_factor' not in model_data:
            raise ValueError("opt_scaling_factor not found in .mat file. Please run lambda_grid_search first or provide --scaling_factor")
        scaling_factor = float(model_data['opt_scaling_factor'].item())
        print(f"Using scaling factor from model: {scaling_factor}")
    
    return model_path, scaling_factor


def evaluate_task(task_name: str, model_dir: str, 
                 optimal_scaling_factor: Optional[float] = None,
                 task_settings: Optional[Dict[str, Any]] = None,
                 save_plots: bool = True) -> Dict[str, float]:
    """
    Evaluate a spiking task on a trained model.
    
    Args:
        task_name: Name of the task ('go_nogo', 'xor', 'mante')
        model_dir: Directory containing the trained model
        optimal_scaling_factor: Override scaling factor
        task_settings: Override task settings
        save_plots: Whether to save visualization plots
        
    Returns:
        Performance metrics dictionary
    """
    # Load model and scaling factor
    model_path, scaling_factor = load_model_and_scaling_factor(model_dir, optimal_scaling_factor)
    
    # Create spiking network adapter
    spiking_rnn = LIFNetworkAdapter(model_path, scaling_factor)
    
    # Create task
    task = SpikingTaskFactory.create_task(task_name, task_settings)
    print(f"Created {task.__class__.__name__} with settings: {task.settings}")
    
    # Evaluate performance
    stimulus, label = task.generate_stimulus()
    performance = task.evaluate_trial(spiking_rnn, stimulus, label)
    
    # Create visualizations if requested
    if save_plots:
        print(f"\nGenerating sample trials and visualizations...")
        results = []
        
        # Generate sample trials using task's sample trial types
        sample_trial_types = task.get_sample_trial_types()
        if sample_trial_types:
            for trial_type in sample_trial_types:
                try:
                    stimulus, label = task.generate_stimulus(trial_type)
                    result = task.evaluate_trial(spiking_rnn, stimulus, label)
                    results.append(result)
                except Exception as e:
                    print(f"Warning: Failed to generate trial type '{trial_type}': {e}")
        else:
            # Fallback: generate a few random trials
            for _ in range(4):
                try:
                    stimulus, label = task.generate_stimulus()
                    result = task.evaluate_trial(spiking_rnn, stimulus, label)
                    results.append(result)
                except Exception as e:
                    print(f"Warning: Failed to generate random trial: {e}")
        
        # Save visualizations using task's built-in methods
        if hasattr(task, 'create_visualization') and results:
            try:
                task.create_visualization(results, model_dir)
                plot_dir = os.path.join(model_dir, 'plots')
                print(f"Plots saved to: {plot_dir}")
            except Exception as e:
                print(f"Warning: Failed to create visualizations: {e}")
        elif results:
            print(f"Generated {len(results)} sample trials (no visualization method available)")
        else:
            print("No sample trials were generated for visualization")
    
    return performance


def main():

    parser = argparse.ArgumentParser(
        description='Evaluate trained spiking RNN models on cognitive tasks.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m spiking.eval_tasks --task go_nogo --model_dir models/go-nogo/
  python -m spiking.eval_tasks --task xor --model_dir models/xor/ --n_trials 50
  python -m spiking.eval_tasks --task mante --model_dir models/mante/ --scaling_factor 45.0
        """
    )
    
    # Get available tasks from factory
    from .tasks import SpikingTaskFactory
    available_tasks = SpikingTaskFactory.list_available_tasks()
    
    parser.add_argument('--task', type=str, required=True,
                       help=f'Task to evaluate. Available: {", ".join(available_tasks)}')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing the trained model .mat file')
    parser.add_argument('--scaling_factor', type=float, default=None,
                       help='Override scaling factor (uses value from .mat file if not provided)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating visualization plots')
    
    # Task-specific settings (advanced usage)
    parser.add_argument('--T', type=int, help='Trial duration (timesteps)')
    parser.add_argument('--stim_on', type=int, help='Stimulus onset time')
    parser.add_argument('--stim_dur', type=int, help='Stimulus duration')
    
    args = parser.parse_args()
    
    # Build task settings from arguments
    task_settings = {}
    for param in ['T', 'stim_on', 'stim_dur']:
        value = getattr(args, param)
        if value is not None:
            task_settings[param] = value
    
    task_settings = task_settings if task_settings else None
    
    try:
        performance = evaluate_task(
            task_name=args.task,
            model_dir=args.model_dir,
            optimal_scaling_factor=args.scaling_factor,
            task_settings=task_settings,
            save_plots=not args.no_plots
        )
        
        print(f"\n✓ Evaluation completed successfully!")
        # print(f"Performance: {performance}")
        return 0
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
    
    # Usage:
    """
    python -m spiking.eval_tasks --task go_nogo --model_dir models/go-nogo/
    python -m spiking.eval_tasks --task xor --model_dir models/xor/ --scaling_factor 45.0
    """