# Rate RNN Package

This package provides continuous-variable rate recurrent neural network (RNN) models based on the framework presented in [this paper](https://www.pnas.org/content/116/45/22811). The rate models can be trained on various cognitive tasks and subsequently mapped to spiking neural networks using the companion `spiking` package.

**Note:** This is a PyTorch conversion of the original TensorFlow implementation. The original TensorFlow code can be found in [https://github.com/rkim35/spikeRNN](https://github.com/rkim35/spikeRNN)

## Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.8.0
- NumPy ≥ 1.16.4
- SciPy ≥ 1.3.1

## Installation

### Installation

```bash
pip install rate
```

### From Source

```bash
git clone https://github.com/NuttidaLab/spikeRNN.git
cd spikeRNN/rate
pip install -e .
```

## Quick Start

```python
import torch
from rate import FR_RNN_dale, set_gpu, create_default_config
from rate.model import generate_input_stim_go_nogo, generate_target_continuous_go_nogo

# Setup
device = set_gpu('0', 0.4)
config = create_default_config(N=200, P_inh=0.2, P_rec=0.2)

# Create network
w_in = torch.randn(200, 1, device=device)
w_out = torch.randn(1, 200, device=device) / 100
net = FR_RNN_dale(200, 0.2, 0.2, w_in, som_N=0, w_dist='gaus',
                  gain=1.5, apply_dale=True, w_out=w_out, device=device)

# Generate task data
settings = {'T': 200, 'stim_on': 50, 'stim_dur': 25, 'DeltaT': 1, 
           'taus': [10], 'task': 'go-nogo'}
u, label = generate_input_stim_go_nogo(settings)
target = generate_target_continuous_go_nogo(settings, label)

# Forward pass
u_tensor = torch.tensor(u, dtype=torch.float32, device=device)
training_params = {'activation': 'sigmoid', 'P_rec': 0.2}
outputs = net.forward(u_tensor, settings['taus'], training_params, settings)
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
- Tests working memory and logic

### Mante Task
Context-dependent sensory integration:
- Multiple sensory modalities
- Context determines relevant modality
- Tests flexible cognitive control

## Training
The main file (`rate/main.py`) takes the following input arguments:

- `gpu` (optional; default `0`): specifies which gpu to use (applicable for a system with a GPU)
- `gpu_frac` (optional; default `0.4`): fraction of available vRAM to allocate
- `n_trials` (required; default `200`): maximum number of trials for training
- `mode` (required; default `train`): either `train` or `eval` (case *insensitive*)
- `output_dir` (required): output directory path where the trained model will be saved. The model will be saved under `<output_dir>/models/<task>`.
- `N` (required): RNN size (i.e. total number of units)
- `gain` (optional; default `1.5`): gain term for the initial connectivity weights
- `P_inh` (optional; default `0.20`): *proportion* of the *N* units that will be inhibitory
- `som_N` (optional; default `0`): *number* of units that will be "somatostatin-expressing" units. Refer to the preprint for more info.
- `apply_dale` (required; default `true`): apply Dale's principle
- `task` (required): task the rate RNN will be trained to perform. Available options are `go-nogo` (Go-NoGo task), `mante` (context-dependent sensory integration task), or `xor` (temporal exclusive OR task). 
- `act` (required; default `sigmoid`): activation function. Available options are `sigmoid`, `clipped_relu`, or `softplus`.
- `loss_fn` (required; default `l2`): loss function (L1, L2, etc...). Case *insensitive*.
- `decay_taus` (required): synaptic decay time-constants (either `a b` for min a and max b or `a` for homogeneous time-constants). Multiply these numbers by 5 to convert to ms. For example, `4 20` means the min and max time constants are 20 ms and 100 ms, respectively.


```bash
python main.py --gpu 0 --gpu_frac 0.20 \
--n_trials 5000 --mode train \
--N 200 --P_inh 0.20 --som_N 0 --apply_dale True \
--gain 1.5 --task go-nogo --act sigmoid --loss_fn l2 \
--decay_taus 4 20 --output_dir ../
```

The trained PyTorch model will be saved as a `.pth` file, and the parameters will be saved as a MATLAB-formatted file (`.mat`) in the output directory.

The name of the output files conform to the following convention:

```
Task_<Task Name>_N_<N>_Taus_<min_tau>_<max_tau>_Act_<act>_<YYYY_MM_DD_TIME>.mat
Task_<Task Name>_N_<N>_Taus_<min_tau>_<max_tau>_Act_<act>_<YYYY_MM_DD_TIME>.pth
```

## API Reference

### Core Classes

- `FR_RNN_dale`: Main firing-rate RNN class with Dale's principle
- `RNNConfig`: Configuration dataclass for network parameters
- `AbstractRateRNN`: Base class for extending RNN implementations

### Task Functions

- `generate_input_stim_go_nogo()`: Generate Go-NoGo task stimuli
- `generate_input_stim_xor()`: Generate XOR task stimuli  
- `generate_input_stim_mante()`: Generate Mante task stimuli
- `generate_target_continuous_*()`: Generate target signals for each task

### Utilities

- `set_gpu()`: Configure GPU usage
- `loss_op()`: Compute training loss
- `eval_rnn()`: Evaluate trained networks

## Citation

If you use this package in your research, please cite the original paper:

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
    Year = {2019}
}
```

## Links

- **Paper**: [PNAS 2019](https://www.pnas.org/content/116/45/22811)
- **Preprint**: [bioRxiv](https://www.biorxiv.org/content/10.1101/579706v2)
- **Documentation**: [Read the Docs](https://rateRNN.readthedocs.io/)
