"""
Base classes and interfaces for spiking neural network tasks.

This module defines abstract base classes and concrete implementations for various
cognitive tasks that can be evaluated using spiking neural networks. Each task
provides methods for stimulus generation, evaluation, and performance analysis.

Classes:
    AbstractSpikingTask: Base abstract class for all spiking tasks
    GoNogoSpikingTask: Go/NoGo task for spiking networks
    XORSpikingTask: XOR task for spiking networks
    ManteSpikingTask: Context-dependent sensory integration task for spiking networks
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import os
from .abstract import AbstractSpikingRNN


class AbstractSpikingTask(ABC):
    """
    Abstract base class for spiking neural network tasks.
    
    This class defines the interface for evaluating spiking networks on cognitive tasks.
    Each task is responsible for generating stimuli, running evaluations, and analyzing
    performance metrics specific to spiking implementations.
    """
    
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """
        Initialize the spiking task with settings.
        
        Args:
            settings (Optional[Dict[str, Any]]): Task-specific settings dictionary.
        """
        self.settings = settings or self.get_default_settings()
        self.validate_settings()
    
    @abstractmethod
    def get_default_settings(self) -> Dict[str, Any]:
        """
        Get default settings for the task.
        
        Returns:
            Dict[str, Any]: Default task settings.
        """
        pass
    
    @abstractmethod
    def validate_settings(self) -> None:
        """
        Validate that all required settings are present and valid.
        
        Raises:
            ValueError: If required settings are missing or invalid.
        """
        pass
    
    @abstractmethod
    def generate_stimulus(self, trial_type: Optional[str] = None) -> Tuple[np.ndarray, Any]:
        """
        Generate input stimulus for the task.
        
        Args:
            trial_type (Optional[str]): Specific trial type to generate.
            
        Returns:
            Tuple[np.ndarray, Any]: Input stimulus array and label/condition.
        """
        pass
    
    @abstractmethod
    def evaluate_trial(self, spiking_rnn: AbstractSpikingRNN, 
                      stimulus: np.ndarray, label: Any) -> Dict[str, Any]:
        """
        Evaluate a single trial on the spiking network.
        
        Args:
            spiking_rnn (AbstractSpikingRNN): Spiking network to evaluate.
            stimulus (np.ndarray): Input stimulus.
            label (Any): Expected label/condition.
            
        Returns:
            Dict[str, Any]: Trial evaluation results.
        """
        pass
    
    @abstractmethod
    def evaluate_performance(self, spiking_rnn: AbstractSpikingRNN, 
                           n_trials: int = 100) -> Dict[str, float]:
        """
        Evaluate performance over multiple trials.
        
        Args:
            spiking_rnn (AbstractSpikingRNN): Spiking network to evaluate.
            n_trials (int): Number of trials to evaluate.
            
        Returns:
            Dict[str, float]: Performance metrics.
        """
        pass
    
    def create_plots_directory(self, base_dir: str) -> str:
        """
        Create directory for saving plots.
        
        Args:
            base_dir (str): Base directory path.
            
        Returns:
            str: Path to plots directory.
        """
        plot_dir = os.path.join(base_dir, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        return plot_dir
    
    def get_sample_trial_types(self) -> List[str]:
        """
        Get sample trial types for visualization.
        
        This method should be overridden by concrete task classes to specify
        what trial types should be used for generating sample visualizations.
        
        Returns:
            List[str]: List of trial type identifiers for this task.
        """
        return []  # Default: no specific trial types


class GoNogoSpikingTask(AbstractSpikingTask):
    """
    Go/NoGo impulse control task for spiking neural networks.
    
    Evaluates the network's ability to respond to "Go" stimuli and withhold responses
    to "NoGo" stimuli using spiking implementations.
    """
    
    def get_default_settings(self) -> Dict[str, Any]:
        """Get default Go/NoGo task settings."""
        return {
            'T': 201,
            'stim_on': 30,
            'stim_dur': 20,
            'delay': 10
        }
    
    def validate_settings(self) -> None:
        """Validate Go/NoGo task settings."""
        required_keys = ['T', 'stim_on', 'stim_dur']
        for key in required_keys:
            if key not in self.settings:
                raise ValueError(f"Missing required setting: {key}")
        
        if self.settings['stim_on'] + self.settings['stim_dur'] >= self.settings['T']:
            raise ValueError("Stimulus extends beyond trial duration")
    
    def get_sample_trial_types(self) -> List[str]:
        """Get sample trial types for Go/NoGo visualization."""
        return ['go', 'nogo']
    
    def generate_stimulus(self, trial_type: Optional[str] = None) -> Tuple[np.ndarray, str]:
        """
        Generate stimulus for Go/NoGo task.
        
        Args:
            trial_type (Optional[str]): 'go' or 'nogo' for specific trial types.
            
        Returns:
            Tuple[np.ndarray, str]: Stimulus and trial type ('go' or 'nogo').
        """
        T = self.settings['T']
        stim_on = self.settings['stim_on']
        stim_dur = self.settings['stim_dur']
        
        u = np.zeros((1, T))
        
        if trial_type is None:
            trial_type = 'go' if np.random.rand() <= 0.5 else 'nogo'
        
        if trial_type == 'go':
            u[0, stim_on:stim_on+stim_dur] = 1
        
        return u, trial_type
    
    def evaluate_trial(self, spiking_rnn: AbstractSpikingRNN, 
                      stimulus: np.ndarray, label: str) -> Dict[str, Any]:
        """
        Evaluate a single Go/NoGo trial.
        
        Args:
            spiking_rnn (AbstractSpikingRNN): Spiking network to evaluate.
            stimulus (np.ndarray): Input stimulus.
            label (str): 'go' or 'nogo'.
            
        Returns:
            Dict[str, Any]: Trial results including spikes, output, and performance.
        """
        # Simulate the network
        stims = {'mode': 'none'}
        spikes, voltages, output, params = spiking_rnn.simulate(stimulus, stims)
        
        # Calculate performance metrics
        output_mean = np.mean(output)
        
        # Determine if response is correct
        if label == 'go':
            correct = output_mean > 0.5  # Should respond
        else:
            correct = output_mean <= 0.5  # Should not respond
        
        return {
            'stimulus': stimulus,
            'label': label,
            'spikes': spikes,
            'voltages': voltages,
            'output': output,
            'output_mean': output_mean,
            'correct': correct,
            'params': params
        }
    
    def evaluate_performance(self, spiking_rnn: AbstractSpikingRNN, 
                           n_trials: int = 100) -> Dict[str, float]:
        """
        Evaluate performance over multiple Go/NoGo trials.
        
        Args:
            spiking_rnn (AbstractSpikingRNN): Spiking network to evaluate.
            n_trials (int): Number of trials to evaluate.
            
        Returns:
            Dict[str, float]: Performance metrics.
        """
        go_correct = 0
        nogo_correct = 0
        go_trials = 0
        nogo_trials = 0
        
        for _ in range(n_trials):
            stimulus, label = self.generate_stimulus()
            result = self.evaluate_trial(spiking_rnn, stimulus, label)
            
            if label == 'go':
                go_trials += 1
                if result['correct']:
                    go_correct += 1
            else:
                nogo_trials += 1
                if result['correct']:
                    nogo_correct += 1
        
        go_accuracy = go_correct / go_trials if go_trials > 0 else 0
        nogo_accuracy = nogo_correct / nogo_trials if nogo_trials > 0 else 0
        overall_accuracy = (go_correct + nogo_correct) / n_trials
        
        return {
            'overall_accuracy': overall_accuracy,
            'go_accuracy': go_accuracy,
            'nogo_accuracy': nogo_accuracy,
            'go_trials': go_trials,
            'nogo_trials': nogo_trials
        }
    
    def create_visualization(self, results: List[Dict[str, Any]], save_dir: str) -> None:
        """
        Create visualization plots for Go/NoGo task results.
        
        Args:
            results (List[Dict[str, Any]]): List of trial results.
            save_dir (str): Directory to save plots.
        """
        plot_dir = self.create_plots_directory(save_dir)
        
        # Separate Go and NoGo trials
        go_results = [r for r in results if r['label'] == 'go']
        nogo_results = [r for r in results if r['label'] == 'nogo']
        
        if go_results:
            self._plot_spike_raster(go_results[0], 'Go', 
                                  os.path.join(plot_dir, 'go_spike_raster.png'))
        
        if nogo_results:
            self._plot_spike_raster(nogo_results[0], 'NoGo', 
                                  os.path.join(plot_dir, 'nogo_spike_raster.png'))
        
        # Plot output comparison
        if go_results and nogo_results:
            self._plot_output_comparison(go_results[0], nogo_results[0], 
                                       os.path.join(plot_dir, 'network_output.png'))
    
    def _plot_spike_raster(self, result: Dict[str, Any], trial_type: str, filename: str) -> None:
        """Plot spike raster for a trial."""
        spikes = result['spikes']
        params = result['params']
        dt = params['dt']
        nt = params['nt']
        
        t = np.arange(dt, dt*(nt+1), dt)[:nt]
        
        plt.figure(figsize=(8, 6))
        
        N = spikes.shape[0]
        for neuron_idx in range(N):
            curr_spk = spikes[neuron_idx, 9:]  # Skip first 9 timepoints
            spike_indices = np.where(curr_spk > 0)[0]
            if len(spike_indices) > 0:
                spike_times = t[9:][spike_indices]
                plt.plot(spike_times, np.ones(len(spike_times)) * neuron_idx, 
                        'r.', markersize=4)
        
        plt.xlim([0, 1])
        plt.ylim([-5, N+5])
        plt.xlabel('Time (s)')
        plt.ylabel('Neuron Index')
        plt.title(f'{trial_type} Spike Raster')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def _plot_output_comparison(self, go_result: Dict[str, Any], 
                              nogo_result: Dict[str, Any], filename: str) -> None:
        """Plot output comparison between Go and NoGo trials."""
        params = go_result['params']
        dt = params['dt']
        nt = params['nt']
        t = np.arange(dt, dt*(nt+1), dt)[:nt]
        
        plt.figure(figsize=(10, 6))
        plt.plot(t, nogo_result['output'].flatten(), 'm', linewidth=2, label='NoGo')
        plt.plot(t, go_result['output'].flatten(), 'g', linewidth=2, label='Go')
        plt.xlabel('Time (s)')
        plt.ylabel('Network Output')
        plt.legend()
        plt.title('Network Output Comparison')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


class XORSpikingTask(AbstractSpikingTask):
    """
    XOR temporal logic task for spiking neural networks.
    
    Evaluates the network's ability to perform XOR logic on temporal sequences
    using spiking implementations.
    """
    
    def get_default_settings(self) -> Dict[str, Any]:
        """Get default XOR task settings."""
        return {
            'T': 400,
            'stim_on': 50,
            'stim_dur': 50,
            'delay': 20
        }
    
    def validate_settings(self) -> None:
        """Validate XOR task settings."""
        required_keys = ['T', 'stim_on', 'stim_dur', 'delay']
        for key in required_keys:
            if key not in self.settings:
                raise ValueError(f"Missing required setting: {key}")
        
        total_stim_time = self.settings['stim_on'] + 2 * self.settings['stim_dur'] + self.settings['delay']
        if total_stim_time >= self.settings['T']:
            raise ValueError("Stimuli extend beyond trial duration")
    
    def get_sample_trial_types(self) -> List[str]:
        """Get sample trial types for XOR visualization."""
        return ['++', '+-', '-+', '--']
    
    def generate_stimulus(self, trial_type: Optional[str] = None) -> Tuple[np.ndarray, str]:
        """
        Generate stimulus for XOR task.
        
        Args:
            trial_type (Optional[str]): Specific pattern ('++', '+-', '-+', '--').
            
        Returns:
            Tuple[np.ndarray, str]: Stimulus and expected output ('same' or 'diff').
        """
        T = self.settings['T']
        stim_on = self.settings['stim_on']
        stim_dur = self.settings['stim_dur']
        delay = self.settings['delay']
        
        u = np.zeros((2, T))
        
        if trial_type is None:
            # Generate random pattern
            patterns = ['++', '+-', '-+', '--']
            trial_type = np.random.choice(patterns)
        
        # Parse pattern
        first_stim = 1 if trial_type[0] == '+' else -1
        second_stim = 1 if trial_type[1] == '+' else -1
        
        u[0, stim_on:stim_on+stim_dur] = first_stim
        u[1, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay] = second_stim
        
        # Determine expected output
        expected = 'same' if first_stim == second_stim else 'diff'
        
        return u, expected
    
    def evaluate_trial(self, spiking_rnn: AbstractSpikingRNN, 
                      stimulus: np.ndarray, label: str) -> Dict[str, Any]:
        """
        Evaluate a single XOR trial.
        
        Args:
            spiking_rnn (AbstractSpikingRNN): Spiking network to evaluate.
            stimulus (np.ndarray): Input stimulus.
            label (str): Expected output ('same' or 'diff').
            
        Returns:
            Dict[str, Any]: Trial results.
        """
        # Simulate the network
        stims = {'mode': 'none'}
        spikes, voltages, output, params = spiking_rnn.simulate(stimulus, stims)
        
        # Analyze output during decision period
        stim_on = self.settings['stim_on']
        stim_dur = self.settings['stim_dur']
        delay = self.settings['delay']
        task_end_T = stim_on + (2 * stim_dur) + delay
        target_onset = 10 + task_end_T
        target_offset = target_onset + 100
        
        if target_offset <= len(output):
            decision_output = np.mean(output[target_onset:target_offset])
        else:
            decision_output = np.mean(output[-50:])  # Use last 50 time points
        
        # Determine predicted response
        predicted = 'same' if decision_output > 0 else 'diff'
        correct = predicted == label
        
        return {
            'stimulus': stimulus,
            'label': label,
            'predicted': predicted,
            'spikes': spikes,
            'voltages': voltages,
            'output': output,
            'decision_output': decision_output,
            'correct': correct,
            'params': params
        }
    
    def evaluate_performance(self, spiking_rnn: AbstractSpikingRNN, 
                           n_trials: int = 1) -> Dict[str, float]:
        """
        Evaluate performance over multiple XOR trials.
        
        Args:
            spiking_rnn (AbstractSpikingRNN): Spiking network to evaluate.
            n_trials (int): Number of trials to evaluate.
            
        Returns:
            Dict[str, float]: Performance metrics.
        """
        correct_trials = 0
        pattern_counts = {'++': 0, '+-': 0, '-+': 0, '--': 0}
        pattern_correct = {'++': 0, '+-': 0, '-+': 0, '--': 0}
        
        for _ in range(n_trials):
            stimulus, label = self.generate_stimulus()
            result = self.evaluate_trial(spiking_rnn, stimulus, label)
            
            # Determine pattern from stimulus
            pattern = self._get_pattern_from_stimulus(stimulus)
            pattern_counts[pattern] += 1
            
            if result['correct']:
                correct_trials += 1
                pattern_correct[pattern] += 1
        
        overall_accuracy = correct_trials / n_trials
        
        # Calculate per-pattern accuracy
        pattern_accuracies = {}
        for pattern in pattern_counts:
            if pattern_counts[pattern] > 0:
                pattern_accuracies[f'{pattern}_accuracy'] = pattern_correct[pattern] / pattern_counts[pattern]
            else:
                pattern_accuracies[f'{pattern}_accuracy'] = 0
        
        results = {
            'overall_accuracy': overall_accuracy,
            **pattern_accuracies,
            **{f'{pattern}_count': pattern_counts[pattern] for pattern in pattern_counts}
        }
        
        return results
    
    def _get_pattern_from_stimulus(self, stimulus: np.ndarray) -> str:
        """Extract pattern from stimulus array."""
        stim_on = self.settings['stim_on']
        stim_dur = self.settings['stim_dur']
        delay = self.settings['delay']
        
        first_val = np.mean(stimulus[0, stim_on:stim_on+stim_dur])
        second_val = np.mean(stimulus[1, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay])
        
        first_char = '+' if first_val > 0 else '-'
        second_char = '+' if second_val > 0 else '-'
        
        return first_char + second_char


class ManteSpikingTask(AbstractSpikingTask):
    """
    Context-dependent sensory integration task for spiking neural networks.
    
    Evaluates the network's ability to perform context-dependent decision making
    using spiking implementations.
    """
    
    def get_default_settings(self) -> Dict[str, Any]:
        """Get default Mante task settings."""
        return {
            'T': 300,
            'stim_on': 50,
            'stim_dur': 100
        }
    
    def validate_settings(self) -> None:
        """Validate Mante task settings."""
        required_keys = ['T', 'stim_on', 'stim_dur']
        for key in required_keys:
            if key not in self.settings:
                raise ValueError(f"Missing required setting: {key}")
        
        if self.settings['stim_on'] + self.settings['stim_dur'] >= self.settings['T']:
            raise ValueError("Stimulus extends beyond trial duration")
    
    def get_sample_trial_types(self) -> List[str]:
        """Get sample trial types for Mante visualization."""
        return ['color', 'motion']
    
    def generate_stimulus(self, trial_type: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Generate stimulus for Mante task.
        
        Args:
            trial_type (Optional[str]): 'color' or 'motion' for specific contexts.
            
        Returns:
            Tuple[np.ndarray, int]: Stimulus and expected decision (+1 or -1).
        """
        T = self.settings['T']
        stim_on = self.settings['stim_on']
        stim_dur = self.settings['stim_dur']
        
        u = np.zeros((4, T))
        
        # Generate sensory inputs
        color_input = 2.5*(np.random.rand()-0.5)  # [-1.25, 1.25]
        motion_input = 2.5*(np.random.rand()-0.5)  # [-1.25, 1.25]
        
        if trial_type is None:
            trial_type = 'color' if np.random.rand() < 0.5 else 'motion'
        
        if trial_type == 'color':
            u[0, stim_on:stim_on+stim_dur] = 1  # context cue
            u[1, stim_on:stim_on+stim_dur] = color_input  # color input
            u[2, stim_on:stim_on+stim_dur] = motion_input  # motion input (irrelevant)
            label = 1 if color_input > 0 else -1
        else:
            u[0, stim_on:stim_on+stim_dur] = -1  # context cue
            u[1, stim_on:stim_on+stim_dur] = color_input  # color input (irrelevant)
            u[2, stim_on:stim_on+stim_dur] = motion_input  # motion input
            label = 1 if motion_input > 0 else -1
        
        return u, label
    
    def evaluate_trial(self, spiking_rnn: AbstractSpikingRNN, 
                      stimulus: np.ndarray, label: int) -> Dict[str, Any]:
        """
        Evaluate a single Mante task trial.
        
        Args:
            spiking_rnn (AbstractSpikingRNN): Spiking network to evaluate.
            stimulus (np.ndarray): Input stimulus.
            label (int): Expected decision (+1 or -1).
            
        Returns:
            Dict[str, Any]: Trial results.
        """
        # Simulate the network
        stims = {'mode': 'none'}
        spikes, voltages, output, params = spiking_rnn.simulate(stimulus, stims)
        
        # Analyze output during decision period
        stim_on = self.settings['stim_on']
        stim_dur = self.settings['stim_dur']
        decision_start = stim_on + stim_dur
        decision_output = np.mean(output[decision_start:])
        
        # Determine predicted decision
        predicted = 1 if decision_output > 0 else -1
        correct = predicted == label
        
        return {
            'stimulus': stimulus,
            'label': label,
            'predicted': predicted,
            'spikes': spikes,
            'voltages': voltages,
            'output': output,
            'decision_output': decision_output,
            'correct': correct,
            'params': params
        }
    
    def evaluate_performance(self, spiking_rnn: AbstractSpikingRNN, 
                           n_trials: int = 100) -> Dict[str, float]:
        """
        Evaluate performance over multiple Mante task trials.
        
        Args:
            spiking_rnn (AbstractSpikingRNN): Spiking network to evaluate.
            n_trials (int): Number of trials to evaluate.
            
        Returns:
            Dict[str, float]: Performance metrics.
        """
        correct_trials = 0
        color_correct = 0
        motion_correct = 0
        color_trials = 0
        motion_trials = 0
        
        for _ in range(n_trials):
            # Alternate between color and motion contexts
            context = 'color' if _ % 2 == 0 else 'motion'
            stimulus, label = self.generate_stimulus(context)
            result = self.evaluate_trial(spiking_rnn, stimulus, label)
            
            if context == 'color':
                color_trials += 1
                if result['correct']:
                    color_correct += 1
            else:
                motion_trials += 1
                if result['correct']:
                    motion_correct += 1
            
            if result['correct']:
                correct_trials += 1
        
        overall_accuracy = correct_trials / n_trials
        color_accuracy = color_correct / color_trials if color_trials > 0 else 0
        motion_accuracy = motion_correct / motion_trials if motion_trials > 0 else 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'color_accuracy': color_accuracy,
            'motion_accuracy': motion_accuracy,
            'color_trials': color_trials,
            'motion_trials': motion_trials
        }


# Task factory for spiking tasks
class SpikingTaskFactory:
    """Factory class for creating spiking task instances."""
    
    _registry = {
        'go_nogo': GoNogoSpikingTask,
        'xor': XORSpikingTask,
        'mante': ManteSpikingTask
    }
    
    @classmethod
    def create_task(cls, task_name: str, settings: Optional[Dict[str, Any]] = None) -> AbstractSpikingTask:
        """
        Create a spiking task instance by type.
        
        Args:
            task_name (str): Name of task ('go_nogo', 'xor', 'mante').
            settings (Optional[Dict[str, Any]]): Task settings.
            
        Returns:
            AbstractSpikingTask: Created task instance.
            
        Raises:
            ValueError: If task type is not recognized.
        """
        if task_name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Task type '{task_name}' not found. Available types: {available}")
        
        task_class = cls._registry[task_name]
        return task_class(settings)
    
    @classmethod
    def register_task(cls, task_name: str, task_class: type) -> None:
        """
        Register a custom task class with the factory.
        
        Args:
            task_name (str): Name to register the task under.
            task_class (type): Task class that inherits from AbstractSpikingTask.
            
        Raises:
            ValueError: If task_class doesn't inherit from AbstractSpikingTask.
        """
        if not issubclass(task_class, AbstractSpikingTask):
            raise ValueError(f"Task class {task_class.__name__} must inherit from AbstractSpikingTask")
        
        cls._registry[task_name] = task_class
        
    @classmethod
    def list_available_tasks(cls) -> list:
        """List all available spiking task types."""
        return list(cls._registry.keys())