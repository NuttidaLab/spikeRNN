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
from spiking.LIF_network_fnc import LIF_network_fnc
from spiking.lambda_grid_search import lambda_grid_search
from spiking.eval_go_nogo import eval_go_nogo

# Optimize scaling factor
lambda_grid_search(
    model_dir='models/go-nogo',
    task_name='go-nogo'
    n_trials=100,
    scaling_factors=list(np.arange(25, 76, 5))
)

# Evaluate performance
eval_go_nogo(
    model_path='models/go-nogo/trained_model.mat'
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
