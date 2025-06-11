# SpikeRNN Framework

This is a PyTorch framework for constructing functional spiking recurrent neural networks from continuous rate models, based on the framework presented in [this paper](https://www.pnas.org/content/116/45/22811)

The SpikeRNN framework consists of two complementary packages:

- **[rate](rate/)**: Continuous-variable rate RNN package for training models on cognitive tasks
- **[spiking](spiking/)**: Spiking RNN package for converting rate models to biologically realistic networks

## Features

### Rate RNN Package
- Implementation of rate-based RNNs with Dale's principle support
- Multiple cognitive task implementations (Go-NoGo, XOR, Mante)

### Spiking RNN Package
- Rate-to-spike conversion maintaining task performance
- Biologically realistic leaky integrate-and-fire (LIF) neurons
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
from spiking import LIF_network_fnc, lambda_grid_search, eval_go_nogo
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

### Complete Workflow

```python
import numpy as np
import torch
import scipy.io as sio
from rate import FR_RNN_dale, set_gpu, create_default_config
from rate.model import generate_input_stim_go_nogo, generate_target_continuous_go_nogo
from spiking import LIF_network_fnc, lambda_grid_search

# Step 1: Set up and train rate RNN
device = set_gpu('0', 0.3)
config = create_default_config(N=200, P_inh=0.2, P_rec=0.2)

# Initialize network
w_in = torch.randn(200, 1, device=device)
w_out = torch.randn(1, 200, device=device) / 100
net = FR_RNN_dale(200, 0.2, 0.2, w_in, som_N=0, w_dist='gaus',
                  gain=1.5, apply_dale=True, w_out=w_out, device=device)

# Generate task data
settings = {'T': 200, 'stim_on': 50, 'stim_dur': 25, 'DeltaT': 1,
           'taus': [10], 'task': 'go-nogo'}
u, label = generate_input_stim_go_nogo(settings)
target = generate_target_continuous_go_nogo(settings, label)

# Train network (simplified)
# ... training loop ...

# Save trained model as .mat for spiking conversion
model_dict = {
    'w': net.w.detach().cpu().numpy(),
    'w_in': net.w_in.detach().cpu().numpy(),
    'w_out': net.w_out.detach().cpu().numpy(),
    'w0': net.w0.detach().cpu().numpy(),
    'N': 200,
    'm': net.m.cpu().numpy(),
    'som_m': net.som_m.cpu().numpy(),
    'inh': net.inh.cpu().numpy(),
    'exc': net.exc.cpu().numpy(),
    'taus': settings['taus'],
    'taus_gaus': net.taus_gaus.detach().cpu().numpy(),
    'taus_gaus0': net.taus_gaus0.detach().cpu().numpy(),
}
sio.savemat('trained_model.mat', model_dict)

# Step 2: Convert to spiking network
scaling_factor = 50.0
u = np.zeros((1, 201))
u[0, 30:50] = 1  # Go trial stimulus

W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(
    'trained_model.mat', scaling_factor, u, {'mode': 'none'},
    downsample=1, use_initial_weights=False
)

print(f"Rate-to-spike conversion completed!")
print(f"Generated {np.sum(spk)} spikes")
print(f"Network output: {out[-1]:.4f}")
```

### Training Rate RNNs

```python
from rate import FR_RNN_dale, set_gpu
from rate.model import generate_input_stim_go_nogo, generate_target_continuous_go_nogo

# Set up device and network
device = set_gpu('0', 0.4)
net = FR_RNN_dale(200, 0.2, 0.2, w_in, som_N=0, w_dist='gaus',
                  gain=1.5, apply_dale=True, w_out=w_out, device=device)

# Generate task data and train
# ... training code ...
```

### Converting to Spiking Networks

```python
from spiking import LIF_network_fnc, lambda_grid_search, eval_go_nogo

# Optimize scaling factor
lambda_grid_search(
    model_path='models/go-nogo/trained_model.mat',
    scaling_range=(20, 80),
    task_type='go-nogo',
    parallel=True
)

# Evaluate performance
eval_go_nogo(
    model_path='models/go-nogo/trained_model.mat',
    scaling_factor=50.0,
    plot_results=True
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


## Examples

### Rate RNN Training Example

```python
import torch
import scipy.io as sio
from rate import FR_RNN_dale, set_gpu
from rate.model import generate_input_stim_go_nogo, generate_target_continuous_go_nogo, loss_op

# Setup
device = set_gpu('0', 0.4)
N = 200
net = FR_RNN_dale(N, 0.2, 0.2, w_in, som_N=0, w_dist='gaus',
                  gain=1.5, apply_dale=True, w_out=w_out, device=device)

# Training loop
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
for trial in range(1000):
    optimizer.zero_grad()
    
    # Generate data
    u, label = generate_input_stim_go_nogo(settings)
    target = generate_target_continuous_go_nogo(settings, label)
    u_tensor = torch.tensor(u, dtype=torch.float32, device=device)
    
    # Forward pass and loss
    outputs = net.forward(u_tensor, settings['taus'], training_params, settings)
    loss = loss_op(outputs, target, training_params)
    
    # Backward pass
    loss.backward()
    optimizer.step()

# Save for spiking conversion
model_dict = {
    'w': net.w.detach().cpu().numpy(),
    'w_in': net.w_in.detach().cpu().numpy(),
    'w_out': net.w_out.detach().cpu().numpy(),
    'w0': net.w0.detach().cpu().numpy(),
    'N': N,
    'm': net.m.cpu().numpy(),
    'som_m': net.som_m.cpu().numpy(),
    'inh': net.inh.cpu().numpy(),
    'exc': net.exc.cpu().numpy(),
    'taus': settings['taus'],
    'taus_gaus': net.taus_gaus.detach().cpu().numpy(),
    'taus_gaus0': net.taus_gaus0.detach().cpu().numpy(),
}
sio.savemat('trained_model.mat', model_dict)
```

### Spiking Network Analysis Example

```python
from spiking import LIF_network_fnc, format_spike_data
import matplotlib.pyplot as plt

# Convert and simulate
W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(
    'trained_model.mat', scaling_factor=50.0, u=stimulus,
    stims={'mode': 'none'}, downsample=1, use_initial_weights=False
)

# Analyze spikes
spike_data = format_spike_data(spk, params['dt'])
print(f"Total spikes: {spike_data['total_spikes']}")
print(f"Mean firing rate: {np.mean(spike_data['firing_rates']):.2f} Hz")

# Plot spike raster
plt.figure(figsize=(10, 6))
spike_times, spike_neurons = np.where(spk)
plt.scatter(spike_times * params['dt'], spike_neurons, s=1, c='black', alpha=0.6)
plt.xlabel('Time (s)')
plt.ylabel('Neuron Index')
plt.title('Spike Raster Plot')
plt.show()
```

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

We welcome contributions! Please see [CONTRIBUTING.md](docs/contributing.rst) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see [LICENCE](LICENCE) file for details.

## Links

- **Documentation**: [Read the Docs](https://spikeRNN.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/NuttidaLab/spikeRNN/issues)
- **Rate Package**: [rate/](rate/)
- **Spiking Package**: [spiking/](spiking/)
