"""
Abstract Base Classes for Rate-based Recurrent Neural Networks

This module defines abstract base classes and interfaces that provide a standardized
framework for implementing rate-based RNN models. These abstractions enable
extensibility and ensure consistent interfaces across different RNN implementations.

Classes:
    AbstractRateRNN: Base class for all rate-based RNN models
    AbstractTaskGenerator: Base class for task stimulus generators
    AbstractTargetGenerator: Base class for target signal generators
    RNNConfig: Configuration dataclass for RNN parameters
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Union, List
import torch
import torch.nn as nn
import numpy as np


@dataclass
class RNNConfig:
    """
    Configuration class for Rate RNN parameters.
    
    Attributes:
        N: Number of neurons in the network
        P_inh: Proportion of inhibitory neurons (0.0 to 1.0)
        P_rec: Recurrent connection probability (0.0 to 1.0)
        som_N: Number of somatostatin-expressing neurons
        gain: Gain factor for weight initialization
        apply_dale: Whether to apply Dale's principle
        w_dist: Weight distribution type ('gaus' or 'gamma')
        device: PyTorch device for computation
    """
    N: int
    P_inh: float
    P_rec: float
    som_N: int = 0
    gain: float = 1.5
    apply_dale: bool = True
    w_dist: str = 'gaus'
    device: str = 'cpu'


class AbstractRateRNN(nn.Module, ABC):
    """
    Abstract base class for rate-based recurrent neural networks.
    
    This class provides the interface that all rate RNN implementations should follow.
    It defines the essential methods that need to be implemented by concrete RNN classes.
    """
    
    def __init__(self, config: RNNConfig) -> None:
        """
        Initialize the abstract RNN with configuration.
        
        Args:
            config (RNNConfig): RNNConfig object containing network parameters.
        """
        super().__init__()
        self.config = config
        self.N = config.N
        self.device = torch.device(config.device)
    
    @abstractmethod
    def assign_exc_inh(self) -> Tuple[np.ndarray, np.ndarray, int, int, Union[np.ndarray, int]]:
        """
        Assign neurons as excitatory or inhibitory.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, int, int, Union[np.ndarray, int]]: Tuple containing:
                - inh: Boolean array marking inhibitory units
                - exc: Boolean array marking excitatory units  
                - NI: Number of inhibitory units
                - NE: Number of excitatory units
                - som_inh: Indices of SOM inhibitory neurons
        """
        pass
    
    @abstractmethod
    def initialize_W(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize the recurrent weight matrix.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
                - w: NxN weight matrix (positive values)
                - mask: NxN sign matrix for Dale's principle
                - som_mask: NxN mask for SOM connectivity constraints
        """
        pass
    
    @abstractmethod
    def forward(self, stim: torch.Tensor, taus: List[float], training_params: Dict[str, Any], 
                settings: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN.
        
        Args:
            stim (torch.Tensor): Input stimulus tensor.
            taus (List[float]): Time constants.
            training_params (Dict[str, Any]): Training configuration parameters.
            settings (Dict[str, Any]): Task-specific settings.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of (outputs, hidden_states).
        """
        pass
    
    @abstractmethod
    def load_net(self, model_dir: str) -> 'AbstractRateRNN':
        """
        Load pre-trained network parameters.
        
        Args:
            model_dir (str): Path to saved model file.
            
        Returns:
            AbstractRateRNN: The loaded network instance.
        """
        pass
    
    def display(self) -> None:
        """
        Display network configuration information.
        This method can be optionally overridden by subclasses.
        """
        print('Abstract Rate RNN Configuration')
        print('====================================')
        print(f'Number of Units: {self.N}')
        print(f'Device: {self.device}')


class AbstractTaskGenerator(ABC):
    """
    Abstract base class for task stimulus generators.
    
    Task generators create input stimuli for different cognitive tasks.
    """
    
    @abstractmethod
    def generate_stimulus(self, settings: Dict[str, Any]) -> Tuple[np.ndarray, Any]:
        """
        Generate task-specific input stimulus.
        
        Args:
            settings (Dict[str, Any]): Task configuration parameters.
            
        Returns:
            Tuple[np.ndarray, Any]: Tuple of (stimulus, labels).
        """
        pass
    
    @abstractmethod
    def get_task_info(self) -> Dict[str, str]:
        """
        Get information about the task.
        
        Returns:
            Dict[str, str]: Dictionary containing task metadata.
        """
        pass


class AbstractTargetGenerator(ABC):
    """
    Abstract base class for target signal generators.
    
    Target generators create desired output signals for training.
    """
    
    @abstractmethod
    def generate_targets(self, settings: Dict[str, Any], labels: Any) -> np.ndarray:
        """
        Generate target output signals.
        
        Args:
            settings (Dict[str, Any]): Task configuration parameters.
            labels (Any): Task labels/conditions.
            
        Returns:
            np.ndarray: Target output array.
        """
        pass


class AbstractLossFunction(ABC):
    """
    Abstract base class for loss functions.
    """
    
    @abstractmethod
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, 
                    training_params: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss between outputs and targets.
        
        Args:
            outputs (torch.Tensor): Network output tensor.
            targets (torch.Tensor): Target tensor.
            training_params (Dict[str, Any]): Training configuration.
            
        Returns:
            torch.Tensor: Loss scalar tensor.
        """
        pass


class RateRNNFactory:
    """
    Factory class for creating rate RNN instances.
    
    This factory enables easy instantiation of different RNN types
    and promotes loose coupling between client code and concrete implementations.
    """
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, rnn_class: type) -> None:
        """
        Register a new RNN class.
        
        Args:
            name (str): String identifier for the RNN type.
            rnn_class (type): The RNN class to register.
        """
        cls._registry[name] = rnn_class
    
    @classmethod
    def create(cls, name: str, config: RNNConfig, **kwargs) -> AbstractRateRNN:
        """
        Create an RNN instance.
        
        Args:
            name (str): String identifier for the RNN type.
            config (RNNConfig): RNN configuration.
            **kwargs: Additional arguments for RNN initialization.
            
        Returns:
            AbstractRateRNN: RNN instance.
            
        Raises:
            ValueError: If the RNN type is not registered.
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown RNN type: {name}")
        
        rnn_class = cls._registry[name]
        return rnn_class(config, **kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all registered RNN types.
        
        Returns:
            List[str]: List of available RNN type names.
        """
        return list(cls._registry.keys())


def validate_config(config: RNNConfig) -> None:
    """
    Validate RNN configuration parameters.
    
    Args:
        config (RNNConfig): RNNConfig object to validate.
        
    Raises:
        ValueError: If configuration parameters are invalid.
    """
    if config.N <= 0:
        raise ValueError("Number of neurons (N) must be positive")
    
    if not 0.0 <= config.P_inh <= 1.0:
        raise ValueError("Inhibitory proportion (P_inh) must be between 0 and 1")
    
    if not 0.0 <= config.P_rec <= 1.0:
        raise ValueError("Connection probability (P_rec) must be between 0 and 1")
    
    if config.som_N < 0:
        raise ValueError("Number of SOM neurons (som_N) must be non-negative")
    
    if config.som_N > config.N:
        raise ValueError("Number of SOM neurons cannot exceed total neurons")
    
    if config.w_dist.lower() not in ['gaus', 'gamma']:
        raise ValueError("Weight distribution must be 'gaus' or 'gamma'")
    
    if config.gain <= 0:
        raise ValueError("Gain must be positive")


def create_default_config(**kwargs) -> RNNConfig:
    """
    Create a default RNN configuration with optional overrides.
    
    Args:
        **kwargs: Parameters to override in the default configuration.
        
    Returns:
        RNNConfig: RNNConfig object with specified or default values.
    """
    defaults = {
        'N': 200,
        'P_inh': 0.2,
        'P_rec': 0.2,
        'som_N': 0,
        'gain': 1.5,
        'apply_dale': True,
        'w_dist': 'gaus',
        'device': 'cpu'
    }
    
    # Override defaults with provided kwargs
    defaults.update(kwargs)
    
    config = RNNConfig(**defaults)
    validate_config(config)
    
    return config 