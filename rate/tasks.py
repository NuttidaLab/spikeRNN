"""
Base classes and interfaces for rate-based RNN tasks.

This module defines abstract base classes and concrete implementations for various
cognitive tasks that can be performed by rate-based RNNs. Each task is responsible
for generating input stimuli and target outputs according to the task specification.

Classes:
    AbstractTask: Base abstract class for all tasks
    GoNogoTask: Go/NoGo impulse control task
    XORTask: XOR temporal logic task  
    ManteTask: Context-dependent sensory integration task
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np


class AbstractTask(ABC):
    """
    Abstract base class for rate-based RNN tasks.
    
    This class defines the interface that all task implementations should follow.
    Each task is responsible for generating input stimuli and target outputs
    according to the specific task requirements.
    """
    
    def __init__(self, settings: Dict[str, Any]):
        """
        Initialize the task with settings.
        
        Args:
            settings (Dict[str, Any]): Task-specific settings dictionary.
        """
        self.settings = settings
        self.validate_settings()
    
    @abstractmethod
    def validate_settings(self) -> None:
        """
        Validate that all required settings are present and valid.
        
        Raises:
            ValueError: If required settings are missing or invalid.
        """
        pass
    
    @abstractmethod
    def generate_stimulus(self, trial_type: Optional[str] = None, seed: Optional[bool] = False) -> Tuple[np.ndarray, Any]:
        """
        Generate input stimulus for the task.
        
        Args:
            trial_type (Optional[str]): Specific trial type to generate (task-dependent).
            seed (Optional[bool]): Whether to use a fixed random seed.
            
        Returns:
            Tuple[np.ndarray, Any]: Input stimulus array and label/condition.
        """
        pass
    
    @abstractmethod
    def generate_target(self, label: Any, seed: Optional[bool] = False) -> np.ndarray:
        """
        Generate target output for the task given a label.
        
        Args:
            label (Any): Task label or condition.
            seed (Optional[bool]): Whether to use a fixed random seed.
            
        Returns:
            np.ndarray: Target output array.
        """
        pass
    
    def simulate_trial(self, trial_type: Optional[str] = None, seed: Optional[bool] = False) -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Simulate a complete trial by generating stimulus and target.
        
        Args:
            trial_type (Optional[str]): Specific trial type to generate (task-dependent).
            seed (Optional[bool]): Whether to use a fixed random seed.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Any]: Stimulus, target, and label.
        """
        stimulus, label = self.generate_stimulus(trial_type=trial_type, seed=seed)
        target = self.generate_target(label, seed=seed)
        return stimulus, target, label


class GoNogoTask(AbstractTask):
    """
    Go/NoGo impulse control task implementation.
    
    In this task, the network must respond to "Go" stimuli and withhold responses
    to "NoGo" stimuli, testing impulse control and decision-making capabilities.
    """
    
    def validate_settings(self) -> None:
        """Validate Go/NoGo task settings."""
        required_keys = ['T', 'stim_on', 'stim_dur']
        for key in required_keys:
            if key not in self.settings:
                raise ValueError(f"Missing required setting: {key}")
        
        if self.settings['stim_on'] + self.settings['stim_dur'] >= self.settings['T']:
            raise ValueError("Stimulus extends beyond trial duration")
    
    def generate_stimulus(self, trial_type: Optional[str] = None, seed: Optional[bool] = False) -> Tuple[np.ndarray, int]:
        """
        Generate the input stimulus matrix for the Go-NoGo task.
        
        Args:
            trial_type (Optional[str]): 'go' or 'nogo' for specific trial types. If None, randomly selected.
            seed (Optional[bool]): Whether to use a fixed random seed.
            
        Returns:
            Tuple[np.ndarray, int]: Tuple containing:
                - u: 1xT stimulus matrix
                - label: Either 1 (Go trial) or 0 (NoGo trial)
        """
        if seed:
            np.random.seed(42)
        
        T = self.settings['T']
        stim_on = self.settings['stim_on']
        stim_dur = self.settings['stim_dur']

        u = np.zeros((1, T))
        
        # Determine trial type
        if trial_type is None:
            # Random selection
            if np.random.rand() <= 0.50:
                trial_type = 'go'
            else:
                trial_type = 'nogo'
        
        # Generate stimulus based on trial type
        if trial_type == 'go':
            u[0, stim_on:stim_on+stim_dur] = 1
            label = 1
        elif trial_type == 'nogo':
            # No stimulus for nogo trials
            label = 0
        else:
            raise ValueError(f"Invalid trial_type '{trial_type}'. Must be 'go', 'nogo', or None.")

        return u, label

    def generate_target(self, label: int, seed: Optional[bool] = False) -> np.ndarray:
        """
        Generate the target output signal for the Go-NoGo task.
        
        Args:
            label (int): Either 1 (Go trial) or 0 (NoGo trial).
            seed (Optional[bool]): Whether to use a fixed random seed.
            
        Returns:
            np.ndarray: 1xT target signal array.
        """
        if seed:
            np.random.seed(42)
        
        T = self.settings['T']
        stim_on = self.settings['stim_on']
        stim_dur = self.settings['stim_dur']

        target = np.zeros((T-1,))
        resp_onset = stim_on + stim_dur
        if label == 1:
            target[resp_onset:] = 1
        else:
            target[resp_onset:] = 0

        return target


class XORTask(AbstractTask):
    """
    XOR temporal logic task implementation.
    
    This task presents two sequential stimuli (+1 or -1) and requires the network
    to output +1 if the stimuli are the same and -1 if they are different.
    """
    
    def validate_settings(self) -> None:
        """Validate XOR task settings."""
        required_keys = ['T', 'stim_on', 'stim_dur', 'delay']
        for key in required_keys:
            if key not in self.settings:
                raise ValueError(f"Missing required setting: {key}")
        
        total_stim_time = self.settings['stim_on'] + 2 * self.settings['stim_dur'] + self.settings['delay']
        if total_stim_time >= self.settings['T']:
            raise ValueError("Stimuli extend beyond trial duration")
    
    def generate_stimulus(self, trial_type: Optional[str] = None, seed: Optional[bool] = False) -> Tuple[np.ndarray, str]:
        """
        Generate the input stimulus matrix for the XOR task.
        
        Args:
            trial_type (Optional[str]): Specific pattern ('++', '+-', '-+', '--'). If None, randomly selected.
            seed (Optional[bool]): Whether to use a fixed random seed.
            
        Returns:
            Tuple[np.ndarray, str]: Tuple containing:
                - u: 2xT stimulus matrix
                - label: Either 'same' or 'diff'
        """
        
        if seed:
            np.random.seed(42)
        
        T = self.settings['T']
        stim_on = self.settings['stim_on']
        stim_dur = self.settings['stim_dur']
        delay = self.settings['delay']

        # Initialize u
        u = np.zeros((2, T))

        # Determine stimulus pattern
        if trial_type is None:
            # Random pattern generation
            labs = []
            if np.random.rand() < 0.50:
                u[0, stim_on:stim_on+stim_dur] = 1
                labs.append(1)
            else:
                u[0, stim_on:stim_on+stim_dur] = -1
                labs.append(-1)

            if np.random.rand() < 0.50:
                u[1, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay] = 1
                labs.append(1)
            else:
                u[1, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay] = -1
                labs.append(-1)

            if np.prod(labs) == 1:
                label = 'same'
            else:
                label = 'diff'
        else:
            # Specific pattern requested
            if trial_type not in ['++', '+-', '-+', '--']:
                raise ValueError(f"Invalid trial_type '{trial_type}'. Must be '++', '+-', '-+', '--', or None.")
            
            # Parse pattern
            first_stim = 1 if trial_type[0] == '+' else -1
            second_stim = 1 if trial_type[1] == '+' else -1
            
            u[0, stim_on:stim_on+stim_dur] = first_stim
            u[1, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay] = second_stim
            
            # Determine expected output
            label = 'same' if first_stim == second_stim else 'diff'

        return u, label

    def generate_target(self, label: str, seed: Optional[bool] = False) -> np.ndarray:
        """
        Generate the target output signal for the XOR task.
        
        Args:
            label (str): Either 'same' or 'diff'.
            seed (Optional[bool]): Whether to use a fixed random seed.
            
        Returns:
            np.ndarray: A 1D target signal array of shape (T-1,).
        """
        if seed:
            np.random.seed(42)
            
        T = self.settings['T']
        stim_on = self.settings['stim_on']
        stim_dur = self.settings['stim_dur']
        delay = self.settings['delay']

        # Calculate the time when the second stimulus presentation ends
        task_end_T = stim_on + (2 * stim_dur) + delay

        # Initialize the target signal array with shape (T-1,)
        z = np.zeros((T - 1,))

        # Define the target window: starts 10 steps after the task ends and lasts for 100 steps
        target_onset = 10 + task_end_T
        target_offset = target_onset + 100

        # Assign the target value based on the label
        if label == 'same':
            z[target_onset:target_offset] = 1
        elif label == 'diff':
            z[target_onset:target_offset] = -1

        return z


class ManteTask(AbstractTask):
    """
    Context-dependent sensory integration task from Mante et al (2013).
    
    This task requires the network to perform context-dependent decision making
    where the relevant sensory modality (color or motion) is determined by a context cue.
    """
    
    def validate_settings(self) -> None:
        """Validate Mante task settings."""
        required_keys = ['T', 'stim_on', 'stim_dur']
        for key in required_keys:
            if key not in self.settings:
                raise ValueError(f"Missing required setting: {key}")
        
        if self.settings['stim_on'] + self.settings['stim_dur'] >= self.settings['T']:
            raise ValueError("Stimulus extends beyond trial duration")
    
    def generate_stimulus(self, trial_type: Optional[str] = None, seed: Optional[bool] = False) -> Tuple[np.ndarray, int]:
        """
        Generate the input stimulus matrix for the sensory integration task.
        
        Args:
            trial_type (Optional[str]): 'color' or 'motion' for specific contexts. If None, randomly selected.
            seed (Optional[bool]): Whether to use a fixed random seed.
            
        Returns:
            Tuple[np.ndarray, int]: Tuple containing:
                - u: 4xT stimulus matrix
                - label: Either +1 or -1 (the correct decision)
        """
        if seed:
            np.random.seed(42)
            
        T = self.settings['T']
        stim_on = self.settings['stim_on']
        stim_dur = self.settings['stim_dur']

        # Initialize stimulus
        u = np.zeros((4, T))

        # Color and motion inputs
        color_input = 2.5*(np.random.rand()-0.5)  # [-1.25, 1.25]
        motion_input = 2.5*(np.random.rand()-0.5)  # [-1.25, 1.25]

        # Determine context
        if trial_type is None:
            # Random context selection
            context = 'color' if np.random.rand() < 0.50 else 'motion'
        else:
            if trial_type not in ['color', 'motion']:
                raise ValueError(f"Invalid trial_type '{trial_type}'. Must be 'color', 'motion', or None.")
            context = trial_type

        # Context signal
        if context == 'color':
            # Context = color task
            u[0, stim_on:stim_on+stim_dur] = 1  # context cue
            u[1, stim_on:stim_on+stim_dur] = color_input  # color input
            u[2, stim_on:stim_on+stim_dur] = motion_input  # motion input (irrelevant in this context)

            if color_input > 0:
                label = 1  # choose option 1
            else:
                label = -1  # choose option 2

        else:
            # Context = motion task
            u[0, stim_on:stim_on+stim_dur] = -1  # context cue
            u[1, stim_on:stim_on+stim_dur] = color_input  # color input (irrelevant in this context)
            u[2, stim_on:stim_on+stim_dur] = motion_input  # motion input

            if motion_input > 0:
                label = 1  # choose option 1
            else:
                label = -1  # choose option 2

        return u, label

    def generate_target(self, label: int, seed: Optional[bool] = False) -> np.ndarray:
        """
        Generate the target output signal for the sensory integration task.
        
        Args:
            label (int): Either +1 or -1, the correct decision.
            seed (Optional[bool]): Whether to use a fixed random seed.
            
        Returns:
            np.ndarray: A 1D target signal array of shape (T-1,).
        """
        T = self.settings['T']
        stim_on = self.settings['stim_on']
        stim_dur = self.settings['stim_dur']

        # Initialize the target signal array with shape (T-1,)
        z = np.zeros((T - 1,))
        
        # Calculate the target onset time dynamically
        target_onset = stim_on + stim_dur

        # Assign the target value from the onset time to the end of the trial
        if label == 1:
            z[target_onset:] = 1
        else:
            z[target_onset:] = -1

        return z


# Task factory for easy instantiation
class TaskFactory:
    """Factory class for creating task instances."""
    
    _registry = {
        'go_nogo': GoNogoTask,
        'xor': XORTask,
        'mante': ManteTask
    }
    
    @classmethod
    def create_task(cls, task_name: str, settings: Dict[str, Any]) -> AbstractTask:
        """
        Create a task instance by type.
        
        Args:
            task_name (str): Name of task ('go_nogo', 'xor', 'mante').
            settings (Dict[str, Any]): Task settings.
            
        Returns:
            AbstractTask: Created task instance.
            
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
        """Register a task class."""
        if not issubclass(task_class, AbstractTask):
            raise ValueError(f"Task class {task_class.__name__} must inherit from AbstractTask")
        cls._registry[task_name] = task_class
    
    
    @classmethod
    def list_available_tasks(cls) -> list:
        """List all available task types."""
        return list(cls._registry.keys())