# SpikeRNN Framework

This is a PyTorch framework for constructing functional spiking recurrent neural networks from continuous rate models, based on the framework presented in [this paper](https://www.pnas.org/content/116/45/22811)

The SpikeRNN framework consists of two complementary packages with a modern **task-based architecture**:

- **[rate](rate/)**: Continuous-variable rate RNN package for training models on cognitive tasks
- **[spiking](spiking/)**: Spiking RNN package for converting rate models to biologically realistic networks

## Features

### Rate RNN Package
- Implementation of rate-based RNNs with Dale's principle support
- Modular task classes (GoNogoTask, XORTask, ManteTask)
- TaskFactory for dynamic task creation
- Multiple cognitive task implementations (Go-NoGo, XOR, Mante)

### Spiking RNN Package
- Rate-to-spike conversion maintaining task performance
- Biologically realistic leaky integrate-and-fire (LIF) neurons
- Spiking task evaluation classes (GoNogoSpikingEvaluator, XORSpikingEvaluator, ManteSpikingEvaluator)
- SpikingEvaluatorFactory for task-based evaluation
- Scaling factor optimization for optimal conversion

## Task-Based Architecture

The framework features a modular task-based design that separates cognitive tasks from models:

- **Easy Extensibility**: Add new tasks without modifying core model code
- **Consistent Interface**: All tasks follow the same abstract interface
- **Factory Pattern**: Dynamic task creation and discovery

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
from spiking import LIF_network_fnc, lambda_grid_search

# Task-based architecture
from rate import TaskFactory
from spiking.eval_tasks import SpikingEvaluatorFactory
from rate.tasks import GoNogoTask, XORTask, ManteTask
from spiking.eval_tasks import GoNogoSpikingEvaluator, XORSpikingEvaluator, ManteSpikingEvaluator
```

## Quick Start: Task-Based Architecture

### Creating and Using Tasks

```python
from rate import TaskFactory

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
from spiking.eval_tasks import SpikingEvaluatorFactory, evaluate_task

# Direct task evaluation (when you have a network instance, not necessarily trained)
spiking_evaluator = SpikingEvaluatorFactory.create_evaluator('go_nogo', settings)
performance = spiking_evaluator.evaluate_single_trial(model_path, scaling_factor)
print(f"Trial result: {performance}")

# High-level interface (when you have model files with trained weights)
performance = evaluate_task(
    task_name='go_nogo',
    model_path='models/go-nogo/model.mat',
    save_plots=True
)

# Command line interface (for scripts and automation)
# python -m spiking.eval_tasks --task go_nogo --model_path models/go-nogo/model.mat
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
from spiking.eval_tasks import SpikingEvaluatorFactory
from rate.tasks import AbstractTask

class MyCustomSpikingEvaluator(AbstractTask):
    def __init__(self, settings):
        super().__init__(settings)
        self.eval_amp_thresh = settings.get('eval_amp_thresh', 0.7) # custom value
    
    def validate_settings(self):
        # Validation logic for custom task
        required_keys = ['T', 'custom_param']
        for key in required_keys:
            if key not in self.settings:
                raise ValueError(f"Missing required setting: {key}")
    
    def evaluate_single_trial(self, model_path: str, scaling_factor: float) -> int:
        """Evaluate a single trial for the custom task."""
        # Custom evaluation logic here
        # Return 1 if correct, 0 if incorrect
        pass

# Register and use with evaluation system
SpikingEvaluatorFactory._registry['my_custom'] = MyCustomSpikingEvaluator

# Now works with eval_tasks.py
python -m spiking.eval_tasks --task my_custom --model_path models/custom/model.mat
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
    model_path='models/go-nogo/model.mat',
    task_name='go-nogo',
    n_trials=100,
    scaling_factors=list(np.arange(25, 76, 5)),
    task_settings=settings
)

# Evaluate performance
performance = evaluate_task(
    task_name='go_nogo',
    model_path='models/go-nogo/model.mat',
    n_trails=50,
    task_settings=settings,
    all_trial_tasks=True
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
@article{Kim_2019,
    Author = {Kim, Robert and Li, Yinghao and Sejnowski, Terrence J.},
    Doi = {10.1073/pnas.1905926116},
    Journal = {Proceedings of the National Academy of Sciences},
    Number = {45},
    Pages = {22811--22820},
    Publisher = {National Academy of Sciences},
    Title = {Simple framework for constructing functional spiking recurrent neural networks},
    Volume = {116},
    Year = {2019}}
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
