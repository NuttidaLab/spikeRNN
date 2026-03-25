#!/usr/bin/env python3
"""
Evaluation script for trained spiking RNN models on cognitive tasks.
"""

import argparse
import os
import sys
import scipy.io as sio
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from .LIF_network_fnc import LIF_network_fnc
from .abstract import AbstractSpikingRNN

# Add the parent directory to sys.path to enable absolute imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from rate.tasks import TaskFactory, AbstractTask, GoNogoTask, XORTask, ManteTask


class LIFNetworkAdapter(AbstractSpikingRNN):
    """
    Adapter to use LIF_network_fnc with the spiking task interface.
    """
    
    def __init__(self, model_path: str, scaling_factor: float):
        from .abstract import SpikingConfig
        config = SpikingConfig(N=200)
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


def load_model_and_scaling_factor(model_path: str, optimal_scaling_factor: Optional[float] = None) -> tuple:
    """
    Load model file and determine scaling factor.
    
    Args:
        model_path: Path to the .mat model file
        optimal_scaling_factor: Override scaling factor if provided
        
    Returns:
        Tuple of (model_path, scaling_factor)
    """
    # Verify the file exists and is a .mat file
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not model_path.endswith('.mat'):
        raise ValueError(f"Expected a .mat file, got: {model_path}")
    
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


def _get_default_task_settings(task_name: str) -> Dict[str, Any]:
    """Get default settings for each task."""
    defaults = {
        'go_nogo': {
            'T': 200,
            'stim_on': 30,
            'stim_dur': 20,
            'eval_amp_thresh': 0.7
            
        },
        'xor': {
            'T': 300,
            'stim_on': 50,
            'stim_dur': 50,
            'delay': 20,
            'eval_amp_thresh': 0.7
        },
        'mante': {
            'T': 300,
            'stim_on': 50,
            'stim_dur': 100,
            'eval_amp_thresh': 0.7
        }
    }
    return defaults.get(task_name, {})


def _get_sample_trial_types(task_name: str) -> list:
    """Get sample trial types for visualization."""
    sample_types = {
        'go_nogo': ['go', 'nogo'],
        'xor': ['++', '+-', '-+', '--'],
        'mante': ['color', 'motion']
    }
    return sample_types.get(task_name, [None])


# Spiking Task Evaluator Classes
# Each rate-task timestep maps to 100 LIF simulation timesteps
_LIF_STEPS_PER_RATE_STEP = 100


class GoNogoSpikingEvaluator(GoNogoTask):
    """Go/NoGo task evaluator for spiking networks."""

    def __init__(self, settings: Dict[str, Any]):
        super().__init__(settings)
        # Add evaluation-specific settings with defaults
        self.eval_amp_thresh = settings.get('eval_amp_thresh', 0.7)
        # Evaluate output after stimulus ends
        self.eval_end = settings.get(
            'eval_end',
            (settings['stim_on'] + settings['stim_dur']) * _LIF_STEPS_PER_RATE_STEP
        )
        
    def evaluate_single_trial(self, model_path: str, scaling_factor: float,
                              model_data: Optional[Dict] = None) -> int:
        """
        Evaluate a single Go/NoGo trial using the original logic.

        Args:
            model_path: Path to the model .mat file
            scaling_factor: Scaling factor for the model
            model_data: Pre-loaded model data dict. If None, loads from model_path.

        Returns:
            int: 1 if trial is correct, 0 if incorrect
        """
        if model_data is None:
            model_data = sio.loadmat(model_path)
        use_initial_weights = False
        down_sample = 1

        try:
            T = self.settings['T']
            stim_on = self.settings['stim_on']
            stim_dur = self.settings['stim_dur']
            eval_amp_thresh = self.eval_amp_thresh

            u = np.zeros((1, T))
            trial_type = 0
            if np.random.rand() >= 0.50:
                u[0, stim_on:stim_on+stim_dur] = 1.0
                trial_type = 1
            stims = {'mode': 'none'}

            W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(model_data, scaling_factor, u, stims, down_sample, use_initial_weights)

            max_output = np.max(out[self.eval_end:])
            
            if trial_type == 1:  # Go trial
                success = max_output > eval_amp_thresh
            else:  # NoGo trial
                success = max_output < 1 - eval_amp_thresh
            
            return 1 if success else 0

        except Exception as e:
            print(f"Error in GoNogoSpikingEvaluator.evaluate_single_trial: {e}")
            return 0


class XORSpikingEvaluator(XORTask):
    """XOR task evaluator for spiking networks."""
    
    def __init__(self, settings: Dict[str, Any]):
        super().__init__(settings)
        # Add evaluation-specific settings with defaults
        self.eval_amp_thresh = settings.get('eval_amp_thresh', 0.7)
        # Evaluate output after second stimulus ends
        self.eval_end = settings.get(
            'eval_end',
            (settings['stim_on'] + 2 * settings['stim_dur'] + settings['delay']) * _LIF_STEPS_PER_RATE_STEP
        )

    def evaluate_single_trial(self, model_path: str, scaling_factor: float,
                              model_data: Optional[Dict] = None) -> int:
        """
        Evaluate a single XOR trial using the original logic.

        Args:
            model_path: Path to the model .mat file
            scaling_factor: Scaling factor for the model
            model_data: Pre-loaded model data dict. If None, loads from model_path.

        Returns:
            int: 1 if trial is correct, 0 if incorrect
        """
        if model_data is None:
            model_data = sio.loadmat(model_path)
        use_initial_weights = False
        down_sample = 1

        try:
            # Use settings from the task instance (inherited from XORTask)
            T = self.settings['T']
            stim_on = self.settings['stim_on']
            stim_dur = self.settings['stim_dur']
            delay = self.settings['delay']
            eval_amp_thresh = self.eval_amp_thresh
            
            u = np.zeros((2, T))
            u_lab = np.zeros(2)
            if np.random.rand() >= 0.5:
                u[0, stim_on:stim_on+stim_dur] = 1
                u_lab[0] = 1
            else:
                u[0, stim_on:stim_on+stim_dur] = -1
                u_lab[0] = -1
            if np.random.rand() >= 0.5:
                u[1, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay] = 1
                u_lab[1] = 1
            else:
                u[1, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay] = -1
                u_lab[1] = -1
            label = np.prod(u_lab)
            stims = {'mode': 'none'}
            _, _, _, _, _, out, _ = LIF_network_fnc(model_data, scaling_factor, u, stims, down_sample, use_initial_weights)
            
            if (label == 1 and np.max(out[self.eval_end:]) > eval_amp_thresh) or (label == -1 and np.min(out[self.eval_end:]) < -eval_amp_thresh):
                return 1
            return 0

        except Exception as e:
            print(f"Error in XORSpikingEvaluator.evaluate_single_trial: {e}")
            return 0


class ManteSpikingEvaluator(ManteTask):
    """Mante task evaluator for spiking networks."""
    
    def __init__(self, settings: Dict[str, Any]):
        super().__init__(settings)
        # Add evaluation-specific settings with defaults
        self.eval_amp_thresh = settings.get('eval_amp_thresh', 0.7)
        # Evaluate output after stimulus ends
        self.eval_end = settings.get(
            'eval_end',
            (settings['stim_on'] + settings['stim_dur']) * _LIF_STEPS_PER_RATE_STEP
        )
        
    def evaluate_single_trial(self, model_path: str, scaling_factor: float,
                              model_data: Optional[Dict] = None) -> int:
        """
        Evaluate a single Mante trial using the original logic.

        Args:
            model_path: Path to the model .mat file
            scaling_factor: Scaling factor for the model
            model_data: Pre-loaded model data dict. If None, loads from model_path.

        Returns:
            int: 1 if trial is correct, 0 if incorrect
        """
        if model_data is None:
            model_data = sio.loadmat(model_path)
        use_initial_weights = False
        down_sample = 1

        try:
            # Use settings from the task instance (inherited from ManteTask)
            T = self.settings['T']
            stim_on = self.settings['stim_on']
            stim_dur = self.settings['stim_dur']
            eval_amp_thresh = self.eval_amp_thresh
            
            u = np.zeros((4, T))
            # Generate sensory inputs (same encoding as rate ManteTask)
            color_input = 2.5 * (np.random.rand() - 0.5)   # [-1.25, 1.25]
            motion_input = 2.5 * (np.random.rand() - 0.5)  # [-1.25, 1.25]

            if np.random.rand() >= 0.5:
                # Color context
                u[0, stim_on:stim_on+stim_dur] = 1       # context cue
                label = 1 if color_input > 0 else -1
            else:
                # Motion context
                u[0, stim_on:stim_on+stim_dur] = -1      # context cue
                label = 1 if motion_input > 0 else -1
            u[1, stim_on:stim_on+stim_dur] = color_input  # color input
            u[2, stim_on:stim_on+stim_dur] = motion_input # motion input
            stims = {'mode': 'none'}
            _, _, _, _, _, out, _ = LIF_network_fnc(model_data, scaling_factor, u, stims, down_sample, use_initial_weights)
            if (label == 1 and np.max(out[self.eval_end:]) > eval_amp_thresh) or (label == -1 and np.min(out[self.eval_end:]) < -eval_amp_thresh):
                return 1
            return 0

        except Exception as e:
            print(f"Error in ManteSpikingEvaluator.evaluate_single_trial: {e}")
            return 0


# Task Evaluator Factory
class SpikingEvaluatorFactory:
    """Factory class for creating spiking task evaluator instances."""
    
    _registry = {
        'go_nogo': GoNogoSpikingEvaluator,
        'xor': XORSpikingEvaluator,
        'mante': ManteSpikingEvaluator
    }
    
    @classmethod
    def create_evaluator(cls, task_name: str, settings: Dict[str, Any]):
        """
        Create a spiking task evaluator instance by type.
        
        Args:
            task_name (str): Name of task ('go_nogo', 'xor', 'mante').
            settings (Dict[str, Any]): Task settings.
            
        Returns:
            Spiking task evaluator instance.
            
        Raises:
            ValueError: If task type is not recognized.
        """
        if task_name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Task type '{task_name}' not found. Available types: {available}")
        
        evaluator_class = cls._registry[task_name]
        return evaluator_class(settings)
    
    @classmethod
    def list_available_tasks(cls) -> list:
        """List all available spiking task evaluator types."""
        return list(cls._registry.keys())


def evaluate_single_trial(task_name: str, model_path: str, scaling_factor: float,
                          settings: Optional[Dict[str, Any]] = None,
                          model_data: Optional[Dict] = None) -> int:
    """
    Evaluate a single trial for a given task using the appropriate evaluator class.

    Args:
        task_name: Name of the task ('go_nogo', 'xor', 'mante')
        model_path: Path to the model .mat file
        scaling_factor: Scaling factor for the model
        settings: Optional custom settings. If None, uses default settings.
        model_data: Pre-loaded model data dict. If None, loads from model_path.

    Returns:
        int: 1 if trial is correct, 0 if incorrect
    """
    try:
        task_name = task_name.replace('-', '_')

        # Use provided settings or get default settings for the task
        if settings is None:
            settings = _get_default_task_settings(task_name)

        # Create the appropriate evaluator using the factory
        evaluator = SpikingEvaluatorFactory.create_evaluator(task_name, settings)

        # Use the evaluator's evaluate_single_trial method
        return evaluator.evaluate_single_trial(model_path, scaling_factor, model_data)
        
    except Exception as e:
        print(f"Error in evaluate_single_trial: {e}")
        return 0


def evaluate_task(task_name: str, model_path: str, 
                 optimal_scaling_factor: Optional[float] = None,
                 task_settings: Optional[Dict[str, Any]] = None,
                 n_trials: int = 100,
                 all_trial_types: bool = False,
                 ) -> Dict[str, float]:
    """
    Evaluate a spiking task on a trained model.
    
    Args:
        task_name: Name of the task ('go_nogo', 'xor', 'mante')
        model_path: Path to the .mat model file
        optimal_scaling_factor: Override scaling factor
        task_settings: Override task settings
        n_trials: Number of trials to evaluate
        all_trial_types: Evaluate all trial types for the task
    
    Returns:
        Performance metrics dictionary
    """
    # Load model and scaling factor
    model_path, scaling_factor = load_model_and_scaling_factor(model_path, optimal_scaling_factor)

    # Create task using rate-based task factory
    task = TaskFactory.create_task(task_name, task_settings or _get_default_task_settings(task_name))
    print(f"Created {task.__class__.__name__} with settings: {task.settings}")

    # Load model data once and reuse across all trials
    model_data = sio.loadmat(model_path)

    results = []
    correct_trials = 0
    incorrect_trials = 0
    # Evaluate single trial performance
    for i in range(n_trials):
        result = evaluate_single_trial(task_name, model_path, scaling_factor, task.settings, model_data)
        results.append(result)
        if result == 1:
            correct_trials += 1
        else:
            incorrect_trials += 1
    
    performance = {'Correct trials': correct_trials,
                   'Incorrect trials': incorrect_trials,
                   'Total trials': len(results)
                   }

    if all_trial_types:
        # Generate all trial types
        sample_trial_types = _get_sample_trial_types(task_name)
        print(f"\nGenerating all trial types: {sample_trial_types}...")

        spiking_rnn = LIFNetworkAdapter(model_path, scaling_factor)
        results = []

        for trial_type in sample_trial_types:
            try:
                stimulus, target, label = task.simulate_trial(trial_type)
                # Simulate the network
                stims = {'mode': 'none'}
                spikes, voltages, output, params = spiking_rnn.simulate(stimulus, stims)
                
                result = {
                    'stimulus': stimulus,
                    'target': target,
                    'label': label,
                    'spikes': spikes,
                    'output': output,
                    'params': params,
                    'trial_type': trial_type
                }
                results.append(result)
                
            except Exception as e:
                print(f"Warning: Failed to generate trial type '{trial_type}': {e}")
        
        pd.DataFrame(results).to_csv(f'{task_name}_all_trial_types.csv')
        
    print(f"Performance: {performance}")
    return performance


def main():

    parser = argparse.ArgumentParser(
        description='Evaluate trained spiking RNN models on cognitive tasks.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m spiking.eval_tasks --task go_nogo --model_path models/go-nogo/model.mat
  python -m spiking.eval_tasks --task xor --model_path models/xor/model.mat --n_trials 50
  python -m spiking.eval_tasks --task mante --model_path models/mante/model.mat --scaling_factor 45.0
        """
    )
    
    # Get available tasks from rate task factory
    available_tasks = TaskFactory.list_available_tasks()
    
    parser.add_argument('--task', type=str, required=True,
                       help=f'Task to evaluate. Available: {", ".join(available_tasks)}')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model .mat file')
    parser.add_argument('--scaling_factor', type=float, default=None,
                       help='Override scaling factor (uses value from .mat file if not provided)')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of trials to evaluate')
    parser.add_argument('--all_trial_types', action='store_true', default=False,
                       help='Generate all trial types for the task')
    
    # Task-specific settings (advanced usage)
    parser.add_argument('--T', type=int, help='Trial duration (timesteps)')
    parser.add_argument('--stim_on', type=int, help='Stimulus onset time')
    parser.add_argument('--stim_dur', type=int, help='Stimulus duration')
    parser.add_argument('--delay', type=int, help='Delay time')
    args = parser.parse_args()
    
    # Build task settings from arguments
    task_settings = {}
    for param in ['T', 'stim_on', 'stim_dur', 'delay']:
        value = getattr(args, param)
        if value is not None:
            task_settings[param] = value
    
    task_settings = task_settings if task_settings else None
    
    try:
        performance = evaluate_task(
            task_name=args.task,
            model_path=args.model_path,
            optimal_scaling_factor=args.scaling_factor,
            task_settings=task_settings,
            n_trials=args.n_trials,
            all_trial_types=args.all_trial_types,
        )
        
        print(f"\n✓ Evaluation completed successfully!")
        # print(f"Results: {results}")
        return 0
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
    
    # Usage:
    """
    python -m spiking.eval_tasks --task go_nogo --model_path models/go-nogo/model.mat
    python -m spiking.eval_tasks --task xor --model_path models/xor/model.mat --scaling_factor 45.0 --n_trials 50
    """