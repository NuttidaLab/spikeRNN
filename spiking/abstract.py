"""
Abstract Base Classes for Spiking Neural Networks

This module defines abstract base classes and interfaces that provide a standardized
framework for implementing spiking neural network models. These abstractions enable
extensibility and ensure consistent interfaces across different spiking implementations.

Classes:
    AbstractSpikingRNN: Base class for all spiking RNN models
    AbstractSpikingConverter: Base class for rate-to-spike converters
    AbstractSpikingEvaluator: Base class for spiking network evaluators
    SpikingConfig: Configuration dataclass for spiking RNN parameters
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Union, List
import torch
import numpy as np


@dataclass
class SpikingConfig:
    """
    Configuration class for Spiking RNN parameters.
    
    Attributes:
        N: Number of neurons in the network
        dt: Integration time step (seconds)
        downsample: Downsampling factor for simulation
        scaling_factor: Scaling factor for rate-to-spike conversion
        tref: Refractory period (seconds)
        tm: Membrane time constant (seconds)
        vreset: Reset voltage (mV)
        vpeak: Spike threshold voltage (mV)
        tr: Synaptic rise time constant (seconds)
        use_initial_weights: Whether to use initial random weights
        device: PyTorch device for computation
    """
    N: int
    dt: float = 0.00005
    downsample: int = 1
    scaling_factor: float = 50.0
    tref: float = 0.002
    tm: float = 0.010
    vreset: float = -65.0
    vpeak: float = -40.0
    tr: float = 0.002
    use_initial_weights: bool = False
    device: str = 'cpu'


class AbstractSpikingRNN(ABC):
    """
    Abstract base class for spiking recurrent neural networks.
    
    This class provides the interface that all spiking RNN implementations should follow.
    It defines the essential methods that need to be implemented by concrete spiking RNN classes.
    """
    
    def __init__(self, config: SpikingConfig) -> None:
        """
        Initialize the abstract spiking RNN with configuration.
        
        Args:
            config (SpikingConfig): SpikingConfig object containing network parameters.
        """
        self.config = config
        self.N = config.N
        self.device = torch.device(config.device)
    
    @abstractmethod
    def load_rate_weights(self, model_path: str) -> None:
        """
        Load weights from a trained rate RNN model.
        
        Args:
            model_path (str): Path to the rate RNN model file.
        """
        pass
    
    @abstractmethod
    def initialize_lif_params(self) -> None:
        """
        Initialize LIF neuron parameters.
        """
        pass
    
    @abstractmethod
    def simulate(self, stimulus: np.ndarray, stims: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
        """
        Simulate the spiking network with given stimulus.
        
        Args:
            stimulus (np.ndarray): Input stimulus array.
            stims (Dict[str, Any]): Stimulation parameters.
            
        Returns:
            Tuple[np.ndarray, ...]: Simulation results including spikes, voltages, outputs.
        """
        pass
    
    @abstractmethod
    def compute_firing_rates(self, spikes: np.ndarray) -> np.ndarray:
        """
        Compute firing rates from spike trains.
        
        Args:
            spikes (np.ndarray): Binary spike matrix.
            
        Returns:
            np.ndarray: Firing rates for each neuron.
        """
        pass


class AbstractSpikingConverter(ABC):
    """
    Abstract base class for rate-to-spike conversion methods.
    
    This class defines the interface for converting rate RNN models to spiking networks.
    """
    
    @abstractmethod
    def convert(self, rate_model_path: str, config: SpikingConfig) -> AbstractSpikingRNN:
        """
        Convert a rate RNN model to a spiking RNN.
        
        Args:
            rate_model_path (str): Path to the rate RNN model.
            config (SpikingConfig): Spiking network configuration.
            
        Returns:
            AbstractSpikingRNN: Converted spiking RNN instance.
        """
        pass
    
    @abstractmethod
    def optimize_scaling_factor(self, rate_model_path: str, task_type: str, 
                              n_trials: int = 100) -> float:
        """
        Optimize the scaling factor for rate-to-spike conversion.
        
        Args:
            rate_model_path (str): Path to the rate RNN model.
            task_type (str): Type of task ('go-nogo', 'xor', 'mante').
            n_trials (int): Number of trials for optimization.
            
        Returns:
            float: Optimal scaling factor.
        """
        pass


class AbstractSpikingEvaluator(ABC):
    """
    Abstract base class for spiking network evaluators.
    
    This class defines the interface for evaluating spiking network performance.
    """
    
    @abstractmethod
    def evaluate_task_performance(self, spiking_rnn: AbstractSpikingRNN, 
                                 task_type: str, n_trials: int = 100) -> Dict[str, float]:
        """
        Evaluate spiking network performance on a specific task.
        
        Args:
            spiking_rnn (AbstractSpikingRNN): Spiking RNN to evaluate.
            task_type (str): Type of task to evaluate.
            n_trials (int): Number of trials for evaluation.
            
        Returns:
            Dict[str, float]: Performance metrics.
        """
        pass
    
    @abstractmethod
    def compare_with_rate_model(self, spiking_rnn: AbstractSpikingRNN, 
                               rate_model_path: str) -> Dict[str, Any]:
        """
        Compare spiking network performance with original rate model.
        
        Args:
            spiking_rnn (AbstractSpikingRNN): Spiking RNN to compare.
            rate_model_path (str): Path to original rate RNN model.
            
        Returns:
            Dict[str, Any]: Comparison results.
        """
        pass
    
    @abstractmethod
    def analyze_spike_dynamics(self, spikes: np.ndarray, dt: float) -> Dict[str, Any]:
        """
        Analyze spike train dynamics and statistics.
        
        Args:
            spikes (np.ndarray): Binary spike matrix.
            dt (float): Time step size.
            
        Returns:
            Dict[str, Any]: Spike dynamics analysis results.
        """
        pass


class SpikingRNNFactory:
    """
    Factory class for creating spiking RNN instances.
    
    This factory enables easy instantiation of different spiking RNN types
    and promotes loose coupling between client code and concrete implementations.
    """
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, spiking_rnn_class: type) -> None:
        """
        Register a spiking RNN class with the factory.
        
        Args:
            name (str): Name to register the class under.
            spiking_rnn_class (type): The spiking RNN class to register.
        """
        cls._registry[name] = spiking_rnn_class
    
    @classmethod
    def create(cls, name: str, config: SpikingConfig, **kwargs) -> AbstractSpikingRNN:
        """
        Create a spiking RNN instance by name.
        
        Args:
            name (str): Name of the registered spiking RNN class.
            config (SpikingConfig): Configuration for the spiking RNN.
            **kwargs: Additional arguments to pass to the constructor.
            
        Returns:
            AbstractSpikingRNN: Created spiking RNN instance.
            
        Raises:
            ValueError: If the requested spiking RNN type is not registered.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Spiking RNN '{name}' not found. Available types: {available}")
        
        spiking_rnn_class = cls._registry[name]
        return spiking_rnn_class(config, **kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available spiking RNN types.
        
        Returns:
            List[str]: List of registered spiking RNN names.
        """
        return list(cls._registry.keys())


def validate_spiking_config(config: SpikingConfig) -> None:
    """
    Validate spiking RNN configuration parameters.
    
    Args:
        config (SpikingConfig): Configuration to validate.
        
    Raises:
        ValueError: If any configuration parameter is invalid.
    """
    if config.N <= 0:
        raise ValueError("Number of neurons (N) must be positive")
    
    if config.dt <= 0:
        raise ValueError("Time step (dt) must be positive")
    
    if config.downsample < 1:
        raise ValueError("Downsample factor must be >= 1")
    
    if config.scaling_factor <= 0:
        raise ValueError("Scaling factor must be positive")
    
    if config.tref < 0:
        raise ValueError("Refractory period (tref) must be non-negative")
    
    if config.tm <= 0:
        raise ValueError("Membrane time constant (tm) must be positive")
    
    if config.tr <= 0:
        raise ValueError("Synaptic rise time (tr) must be positive")
    
    if config.vpeak <= config.vreset:
        raise ValueError("Spike threshold (vpeak) must be greater than reset voltage (vreset)")


def create_default_spiking_config(**kwargs) -> SpikingConfig:
    """
    Create a default spiking RNN configuration.
    
    Args:
        **kwargs: Override default parameters.
        
    Returns:
        SpikingConfig: Default configuration with any overrides applied.
    """
    defaults = {
        'N': 200,
        'dt': 0.00005,
        'downsample': 1,
        'scaling_factor': 50.0,
        'tref': 0.002,
        'tm': 0.010,
        'vreset': -65.0,
        'vpeak': -40.0,
        'tr': 0.002,
        'use_initial_weights': False,
        'device': 'cpu'
    }
    
    # Override defaults with any provided kwargs
    defaults.update(kwargs)
    
    config = SpikingConfig(**defaults)
    validate_spiking_config(config)
    
    return config 