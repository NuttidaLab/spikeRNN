# SpikeRNN Framework

This is a PyTorch framework for constructing functional spiking recurrent neural networks from continuous rate models, based on the framework presented in [this paper](https://www.pnas.org/content/116/45/22811)

The SpikeRNN framework consists of two complementary packages with a modern **task-based architecture**:

- **[rate](rate/)**: Continuous-variable rate RNN package for training models on cognitive tasks
- **[spiking](spiking/)**: Spiking RNN package for converting rate models to biologically realistic networks

## ✨ Task-Based Architecture

The framework now features a modular task-based design that separates cognitive tasks from neural network models:

- **Separation of Concerns**: Tasks and models are independent entities
- **Easy Extensibility**: Add new tasks without modifying core model code
- **Consistent Interface**: All tasks follow the same abstract interface
- **Factory Pattern**: Dynamic task creation and discovery

## Features

### Rate RNN Package
- Implementation of rate-based RNNs with Dale's principle support
- Modular task classes (GoNogoTask, XORTask, ManteTask)
- TaskFactory for dynamic task creation
- Multiple cognitive task implementations (Go-NoGo, XOR, Mante)

### Spiking RNN Package
- Rate-to-spike conversion maintaining task performance
- Biologically realistic leaky integrate-and-fire (LIF) neurons
- Spiking task evaluation classes (GoNogoSpikingTask, XORSpikingTask, ManteSpikingTask)
- SpikingTaskFactory for task-based evaluation
- Scaling factor optimization for optimal conversion

## Installation

Install both rate and spiking packages:

```bash
git clone https://github.com/NuttidaLab/spikeRNN.git
cd spikeRNN
pip install -e .
```

After installation, you can import both packages:

```python
from rate import FR_RNN_dale, set_gpu, create_default_config
from spiking import LIF_network_fnc, lambda_grid_search, evaluate_task

# Task-based architecture
from rate import TaskFactory
from spiking import SpikingTaskFactory
from rate.tasks import GoNogoTask, XORTask, ManteTask
from spiking.tasks import GoNogoSpikingTask, XORSpikingTask, ManteSpikingTask
```

## Quick Start: Task-Based Architecture

### Creating and Using Tasks

```python
from spikeRNN import TaskFactory

# Create task settings
settings = {
    'T': 200,          # Trial duration
    'stim_on': 50,     # Stimulus onset
    'stim_dur': 25,    # Stimulus duration
    'DeltaT': 1        # Sampling rate
}

# Create a Go/NoGo task using the factory
task = TaskFactory.create_task('go_nogo', settings)

# Generate a complete trial (stimulus + target + label)
stimulus, target, label = task.simulate_trial()
print(f"Generated {task.__class__.__name__} trial with label: {label}")
```

### Evaluating Spiking Networks

The framework provides 2 levels of evaluation:

```python
from spiking import SpikingTaskFactory, evaluate_task

# Direct task evaluation (when you have a network instance, not necessarily trained)
spiking_task = SpikingTaskFactory.create_task('go_nogo')
performance = spiking_task.evaluate_performance(spiking_rnn, n_trials=100)
print(f"Accuracy: {performance['overall_accuracy']:.2f}")

# High-level interface (when you have model files with trained weights)
performance = evaluate_task(
    task_name='go_nogo',
    model_dir='models/go-nogo',
    n_trials=100,
    save_plots=True
)

# Command line interface (for scripts and automation)
# python -m spiking.eval_tasks --task go_nogo --model_dir models/go-nogo/
```

### Extending with Custom Tasks

Create custom rate-based tasks:

```python
from rate.tasks import AbstractTask

class MyCustomTask(AbstractTask):
    def validate_settings(self):
        # Validate required settings
        pass
    
    def generate_stimulus(self, trial_type=None, seed=False):
        # Generate custom stimulus
        return stimulus, label
    
    def generate_target(self, label, seed=False):
        # Generate custom target
        return target

# Use your custom task
custom_task = MyCustomTask(settings)
stimulus, target, label = custom_task.simulate_trial()
```

Create custom spiking evaluation tasks:

```python
from spiking.tasks import AbstractSpikingTask, SpikingTaskFactory

class MyCustomSpikingTask(AbstractSpikingTask):
    def get_default_settings(self):
        return {'T': 200, 'custom_param': 1.0}
    
    def get_sample_trial_types(self):
        return ['type_a', 'type_b']  # For visualization
    
    def generate_stimulus(self, trial_type=None):
        # Generate stimulus logic
        return stimulus, label
    
    def evaluate_performance(self, spiking_rnn, n_trials=100):
        # Multi-trial performance evaluation
        return {'accuracy': 0.85, 'n_trials': n_trials}

# Register and use with evaluation system
SpikingTaskFactory.register_task('my_custom', MyCustomSpikingTask)

# Now works with eval_tasks.py
python -m spiking.eval_tasks --task my_custom --model_dir models/custom/
```

## Requirements

- Python >= 3.7
- PyTorch >= 1.8.0
- NumPy >= 1.16.4
- SciPy >= 1.3.1
- Matplotlib >= 3.0.0


## Workflow

1. **Train Rate RNN**: Use the `rate` package to train continuous-variable RNNs on cognitive tasks
2. **Save as .mat**: Export trained model in MATLAB format with all required parameters
3. **Optimize Scaling**: Use `lambda_grid_search()` to find optimal rate-to-spike conversion parameters
4. **Convert to Spiking**: Use `LIF_network_fnc()` to convert to biologically realistic spiking networks
5. **Evaluate Performance**: Compare spiking and rate network performance on tasks
6. **Analyze Dynamics**: Use spike analysis tools to study neural dynamics


## Quick Start

### Training Rate RNNs

```python
from rate import FR_RNN_dale, set_gpu
from rate import TaskFactory

# Set up device and network
device = set_gpu('0', 0.4)
net = FR_RNN_dale(200, 0.2, 0.2, w_in, som_N=0, w_dist='gaus',
                  gain=1.5, apply_dale=True, w_out=w_out, device=device)

# Use task-based API to generate data
settings = {'T': 200, 'stim_on': 50, 'stim_dur': 25, 'DeltaT': 1}
task = TaskFactory.create_task('go_nogo', settings)
u, target, label = task.simulate_trial()

# ... training code ...
```

### Converting to Spiking Networks

```python
from spiking import LIF_network_fnc, lambda_grid_search
from spiking.eval_tasks import evaluate_task
import numpy as np

# Optimize scaling factor
lambda_grid_search(
    model_dir='models/go-nogo',
    task_name='go-nogo',
    n_trials=100,
    scaling_factors=list(np.arange(25, 76, 5))
)

# Evaluate performance
performance = evaluate_task(
    task_name='go_nogo',
    model_dir='models/go-nogo/',
    n_trials=100
)
```

## Supported Tasks

### Go-NoGo Task
Binary decision task requiring response inhibition:
- **Go trials**: Respond to stimulus
- **NoGo trials**: Withhold response
- Tests impulse control and decision-making

### XOR Task  
Temporal exclusive OR requiring working memory:
- Two sequential binary inputs
- Output XOR result after delay
- Tests working memory and logic operations

### Mante Task
Context-dependent sensory integration:
- Multiple sensory modalities
- Context determines relevant modality
- Tests flexible cognitive control


## Model File Requirements

### Rate Package Output
The rate package save models in two formats:
- `.pth` files: PyTorch format for rate model continuation
- `.mat` files: MATLAB format containing all parameters used for spiking conversion

### Spiking Package Input

- Complete connectivity masks (`m`, `som_m`)
- Neuron type assignments (`inh`, `exc`)
- Time constant parameters (`taus`, `taus_gaus`)
- Initial weight states (`w0`)
- Network size and architecture (`N`)


## Citation

If you use this framework in your research, please cite:

```bibtex
@article{kim2019neural,
  title={Neural population dynamics underlying motor learning transfer},
  author={Kim, T. D. and Lian, T. and Yang, G. R.},
  journal={Neuron},
  volume={103},
  number={2},
  pages={355--371},
  year={2019},
  publisher={Elsevier}
}
```

## Contributing

We welcome contributions! Please see [Contributing to SpikeRNN](https://nuttidalab.github.io/spikeRNN/contributing.html) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see [LICENCE](LICENCE) file for details.

## Links

- **Documentation**: [Read the Docs](https://nuttidalab.github.io/spikeRNN/)
- **Issues**: [GitHub Issues](https://github.com/NuttidaLab/spikeRNN/issues)
- **Rate Package**: [rate/](rate/)
- **Spiking Package**: [spiking/](spiking/)
