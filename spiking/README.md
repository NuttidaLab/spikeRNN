# Spiking RNN Package

This package provides leaky integrate-and-fire (LIF) spiking neural networks constructed by mapping pre-trained continuous rate recurrent neural networks (RNNs) based on the framework presented in [this paper](https://www.pnas.org/content/116/45/22811).

**Note:** This is a PyTorch conversion of the original TensorFlow implementation. The original TensorFlow code can be found in [https://github.com/rkim35/spikeRNN](https://github.com/rkim35/spikeRNN)

## Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.8.0
- NumPy ≥ 1.16.4
- SciPy ≥ 1.3.1
- Matplotlib ≥ 3.0.0

## Installation

```bash
pip install spiking
```

### From Source

```bash
git clone https://github.com/NuttidaLab/spikeRNN.git
cd spikeRNN/spiking
pip install -e .
```

### With Rate Package

For the complete framework:

```bash
git clone https://github.com/NuttidaLab/spikeRNN.git
cd spikeRNN

# Install both packages
cd rate && pip install -e . && cd ..
cd spiking && pip install -e . && cd ..
```

## Quick Start

### Basic Rate-to-Spike Conversion

```python
import numpy as np
from spiking import LIF_network_fnc

# Convert trained rate model to spiking network
model_path = 'path/to/trained/model.mat'  # From rate package
scaling_factor = 50.0

# Create Go trial stimulus  
u = np.zeros((1, 201))
u[0, 30:50] = 1  # 20ms stimulus pulse

# Convert and simulate
stims = {'mode': 'none'}
W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(
    model_path, scaling_factor, u, stims, 
    downsample=1, use_initial_weights=False
)

print(f"Generated {np.sum(spk)} spikes")
print(f"Network output: {out[-1]:.4f}")
```

### Scaling Factor Optimization

```python
from spiking import lambda_grid_search

# Find optimal scaling factor for your model
lambda_grid_search(
    model_path='models/go-nogo/model.mat',
    scaling_range=(20, 80),
    n_trials_per_factor=50,
    task_type='go-nogo',
    parallel=True
)
```

### Performance Evaluation

```python
from spiking import eval_go_nogo

# Evaluate spiking network performance
eval_go_nogo(
    model_path='models/go-nogo/model.mat',
    scaling_factor=50.0,
    n_trials=100,
    plot_results=True
)
```

## Core Functions

### LIF_network_fnc()

Main function for rate-to-spike conversion and simulation.

**Parameters:**
- `model_path`: Path to trained rate model (.mat)
- `scaling_factor`: Scaling factor for conversion (20-100 typical range)
- `u`: Input stimulus array (n_inputs × n_timesteps)
- `stims`: Stimulation parameters dictionary
- `downsample`: Temporal downsampling factor
- `use_initial_weights`: Use random instead of trained weights

**Returns:**
- `W`: Scaled recurrent weight matrix
- `REC`: Membrane voltage traces  
- `spk`: Binary spike matrix (neurons × timesteps)
- `rs`: Instantaneous firing rates
- `all_fr`: Average firing rates
- `out`: Network output signal
- `params`: Simulation parameters


### lambda_grid_search()
Grid search optimization for finding optimal scaling factors.

**Parameters:**
- `model_path`: Path to rate model (.mat file)
- `scaling_range`: Range of scaling factors to test
- `n_trials_per_factor`: Trials per scaling factor
- `task_type`: Task type ('go-nogo', 'xor', 'mante')
- `parallel`: Enable parallel processing

### eval_go_nogo()

Evaluate Go-NoGo task performance with visualization.

**Parameters:**
- `model_path`: Path to trained model (.mat file)
- `scaling_factor`: Scaling factor (if known)
- `n_trials`: Number of evaluation trials
- `plot_results`: Generate visualization plots

## Supported Model Format

### MATLAB Models (.mat)

The package requires MATLAB .mat files containing specific parameters for spiking conversion:

```python
# Required .mat file contents:
model_data = {
    'w': recurrent_weights,          # NxN weight matrix
    'w_in': input_weights,           # Nx1 input weights
    'w_out': output_weights,         # 1xN output weights
    'w0': initial_weights,           # NxN initial weights
    'N': network_size,               # Number of neurons
    'm': connectivity_mask,          # NxN connectivity mask
    'som_m': som_mask,              # NxN SOM connectivity mask
    'inh': inhibitory_indices,       # Boolean array for inhibitory neurons
    'exc': excitatory_indices,       # Boolean array for excitatory neurons
    'taus': time_constants,          # Synaptic time constants
    'taus_gaus': gaussian_taus,      # Gaussian time constants
    'taus_gaus0': initial_taus,      # Initial time constants
    # ... other parameters
}
```

### Loading Models

```python
from spiking import load_rate_model

# Load .mat model
model_data = load_rate_model('trained/model/path.mat')

# Validate required parameters
required_keys = ['w', 'w_in', 'w_out', 'N', 'inh', 'exc', 'taus']
missing = [k for k in required_keys if k not in model_data]
if missing:
    print(f"Warning: Missing parameters {missing}")
```

## Scaling Factor Guidelines

The scaling factor controls the rate-to-spike conversion intensity:

- **Low values (20-40)**: Sparse spiking, may lose information
- **Medium values (40-60)**: Balanced spiking, good performance  
- **High values (60-100)**: Dense spiking, may introduce noise

**Recommendations:**
- Start with 50.0 for initial testing
- Use `lambda_grid_search()` for optimization
- Task complexity may require different ranges

## Spike Analysis Tools

### Basic Spike Statistics

```python
from spiking import format_spike_data

# Analyze spike trains
spike_data = format_spike_data(spk, dt=0.00005)

print(f"Total spikes: {spike_data['total_spikes']}")
print(f"Active neurons: {len(spike_data['active_neurons'])}")
print(f"Mean firing rate: {np.mean(spike_data['firing_rates']):.2f} Hz")
```

### Spike Raster Visualization

```python
import matplotlib.pyplot as plt

# Plot spike raster
spike_times, spike_neurons = np.where(spk)
plt.figure(figsize=(10, 6))
plt.scatter(spike_times * dt, spike_neurons, s=1, c='black', alpha=0.6)
plt.xlabel('Time (s)')
plt.ylabel('Neuron Index')
plt.title('Spike Raster Plot')
plt.show()
```

## Tasks

### Go-NoGo Task

Binary decision task with response inhibition:

```python
# Go trial: respond to stimulus
# NoGo trial: withhold response

# Typical stimulus pattern
u_go = np.zeros((1, 201))
u_go[0, 30:50] = 1      # Go stimulus

u_nogo = np.zeros((1, 201))  
u_nogo[0, 30:50] = -1   # NoGo stimulus
```

### XOR Task

Temporal exclusive OR task:

```python
# Two sequential inputs, XOR output after delay
# Requires working memory to combine inputs

# XOR stimulus patterns (input1, input2, expected_output)
patterns = [
    ([1, 0], [0, 1], 1),  # XOR = 1  
    ([1, 0], [1, 0], 0),  # XOR = 0
    ([0, 1], [0, 1], 0),  # XOR = 0
    ([0, 1], [1, 0], 1),  # XOR = 1
]
```

### Mante Task

Context-dependent sensory integration:

```python
# Multiple sensory modalities with context cue
# Context determines which modality to integrate

# Context 1: integrate motion coherence
# Context 2: integrate color coherence  
```

## Performance Metrics

### Conversion Quality

- **Spike Rate**: Total spikes per second
- **Firing Rate Distribution**: Per-neuron firing rates
- **Output Correlation**: Similarity to rate model output
- **Task Accuracy**: Performance on cognitive task

### Task-Specific Metrics

**Go-NoGo:**
- Go trial accuracy (correct responses)
- NoGo trial accuracy (correct inhibition)
- Response time distribution

**XOR:**
- Logic accuracy for all input combinations
- Temporal precision of output

**Mante:**
- Context-dependent accuracy
- Sensory integration performance

## Integration with Rate Package

Complete workflow example:

```python
# Step 1: Train rate RNN (rate package)
import torch
from rate import FR_RNN_dale, set_gpu

device = set_gpu('0', 0.3)
# ... train rate model and save as .mat file ...

# Step 2: Convert to spiking network (spiking package)  
from spiking import LIF_network_fnc, lambda_grid_search

# Optimize scaling factor
lambda_grid_search(model_path='path/to/trained/model.mat')

# Convert and simulate
W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(
    'path/to/trained/model.mat', scaling_factor=50.0, u=stimulus, 
    stims={'mode': 'none'}, downsample=1, use_initial_weights=False
)
```

## API Reference

### Core Functions

- `LIF_network_fnc()`: Rate-to-spike conversion and simulation
- `lambda_grid_search()`: Scaling factor optimization  
- `eval_go_nogo()`: Go-NoGo task evaluation

### Utility Functions

- `load_rate_model()`: Load MATLAB .mat models
- `format_spike_data()`: Format spike data for analysis
- `validate_stimulus()`: Validate input stimuli
- `check_gpu_availability()`: GPU availability check

### Configuration

- `SpikingConfig`: Configuration dataclass
- `create_default_spiking_config()`: Default configuration
- `AbstractSpikingRNN`: Base class for extensions

## Links

- **GitHub**: [https://github.com/NuttidaLab/spikeRNN](https://github.com/NuttidaLab/spikeRNN)
- **Documentation**: [Read the Docs](https://spikeRNN.readthedocs.io/)
- **Rate RNN Package**: [rate](../rate/) 