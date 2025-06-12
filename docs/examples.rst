Examples
========

This section provides detailed examples of using the complete SpikeRNN framework for training rate RNNs and converting them to spiking networks.

Complete Workflow Example
----------------------------------------

This example shows the complete workflow from training a rate RNN to converting it to a spiking network:

.. code-block:: python

    import torch
    import numpy as np
    from rate import FR_RNN_dale, set_gpu, create_default_config
    from rate.model import generate_input_stim_go_nogo, generate_target_continuous_go_nogo
    from spiking import LIF_network_fnc, lambda_grid_search, eval_go_nogo

    # Step 1: Set up GPU and configuration
    device = set_gpu('0', 0.3)
    config = create_default_config(N=200, P_inh=0.2, P_rec=0.2)

    # Step 2: Train rate RNN (simplified example)
    # ... training code ...

    # Step 3: Save trained model
    # Save trained model in .mat format for spiking conversion
    import scipy.io as sio

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

    # Step 4: Convert to spiking network
    scaling_factor = 50.0
    u = np.zeros((1, 201))
    u[0, 30:50] = 1  # Go trial stimulus

    W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(
        'trained_model.mat', scaling_factor, u, {'mode': 'none'},
        downsample=1, use_initial_weights=False
    )

    print(f"Rate-to-spike conversion completed!")
    print(f"Generated {np.sum(spk)} spikes")

Rate RNN Training Examples
----------------------------------------

Go-NoGo Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Go-NoGo task trains the network to respond to "Go" stimuli and withhold responses to "NoGo" stimuli:

.. code-block:: python

    import torch
    import torch.optim as optim
    from rate import FR_RNN_dale, set_gpu
    from rate.model import (generate_input_stim_go_nogo, 
                           generate_target_continuous_go_nogo, loss_op)
    
    # Setup
    device = set_gpu('0', 0.4)
    N = 200
    P_inh = 0.2
    P_rec = 0.2
    
    # Network initialization
    w_in = torch.randn(N, 1, device=device)
    w_out = torch.randn(1, N, device=device) / 100
    net = FR_RNN_dale(N, P_inh, P_rec, w_in, som_N=0, w_dist='gaus',
                      gain=1.5, apply_dale=True, w_out=w_out, device=device)
    
    # Task settings
    settings = {
        'T': 200, 'stim_on': 50, 'stim_dur': 25,
        'DeltaT': 1, 'taus': [10], 'task': 'go-nogo'
    }
    
    training_params = {
        'learning_rate': 0.01, 'loss_threshold': 7,
        'eval_freq': 100, 'P_rec': 0.20, 'activation': 'sigmoid'
    }
    
    # Training loop
    optimizer = optim.Adam(net.parameters(), lr=training_params['learning_rate'])
    n_trials = 1000
    
    for tr in range(n_trials):
        optimizer.zero_grad()
        
        # Generate task data
        u, label = generate_input_stim_go_nogo(settings)
        target = generate_target_continuous_go_nogo(settings, label)
        u_tensor = torch.tensor(u, dtype=torch.float32, device=device)
        
        # Forward pass
        outputs = net.forward(u_tensor, settings['taus'], training_params, settings)
        
        # Compute loss and update
        loss = loss_op(outputs, target, training_params)
        loss.backward()
        optimizer.step()
        
        if tr % 100 == 0:
            print(f"Trial {tr}, Loss: {loss.item():.4f}")

XOR Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The XOR task requires temporal working memory to compute XOR of two sequential inputs:

.. code-block:: python

    from rate.model import generate_input_stim_xor, generate_target_continuous_xor
    
    # Task settings
    settings = {
        'T': 300, 'stim_on': 50, 'stim_dur': 50, 'delay': 10,
        'DeltaT': 1, 'taus': [10], 'task': 'xor'
    }
    
    # Network with 2 inputs for XOR
    w_in = torch.randn(N, 2, device=device)
    net = FR_RNN_dale(N, P_inh, P_rec, w_in, som_N=0, w_dist='gaus',
                      gain=1.5, apply_dale=True, w_out=w_out, device=device)
    
    # Training loop
    for tr in range(n_trials):
        optimizer.zero_grad()
        
        u, label = generate_input_stim_xor(settings)
        target = generate_target_continuous_xor(settings, label)
        u_tensor = torch.tensor(u, dtype=torch.float32, device=device)
        
        outputs = net.forward(u_tensor, settings['taus'], training_params, settings)
        loss = loss_op(outputs, target, training_params)
        loss.backward()
        optimizer.step()

Spiking Network Examples
----------------------------------------

Basic Rate-to-Spike Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert a trained rate RNN to a spiking network:

.. code-block:: python

    from spiking import LIF_network_fnc
    import numpy as np
    import matplotlib.pyplot as plt

    # Load trained model (.mat files only for spiking conversion)
    model_path = 'trained_model.mat'
    scaling_factor = 50.0
    
    # Create test stimulus
    u = np.zeros((1, 201))
    u[0, 30:50] = 1  # Go trial stimulus
    
    # Convert to spiking network
    stims = {'mode': 'none'}
    W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(
        model_path, scaling_factor, u, stims,
        downsample=1, use_initial_weights=False
    )
    
    print(f"Conversion completed!")
    print(f"Generated {np.sum(spk)} spikes")

Scaling Factor Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finding the optimal scaling factor is crucial for good performance:

.. code-block:: python

    from spiking import lambda_grid_search
    
    # Comprehensive grid search
    lambda_grid_search(
        model_path='models/go-nogo/trained_model.mat',
        scaling_range=(20, 100),
        n_trials_per_factor=100,
        task_type='go-nogo',
        parallel=True
    )

Task Performance Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate converted spiking networks:

.. code-block:: python

    from spiking import eval_go_nogo
    
    # Evaluate Go-NoGo performance
    eval_go_nogo(
        model_path='models/go-nogo/trained_model.mat',
        scaling_factor=50.0,
        n_trials=200,
        plot_results=True
    )

Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Process multiple models:

.. code-block:: python

    import os
    from spiking import LIF_network_fnc, lambda_grid_search
    
    # Process all .mat models in directory
    model_dir = 'models/'
    model_paths = [
        'models/go-nogo/trained_model.mat',
        'models/xor/trained_model.mat',
        'models/mante/trained_model.mat'
    ]
    
    results = {}
    for model_path in model_paths:
        if os.path.exists(model_path):
            # Find optimal scaling
            lambda_grid_search(model_path=model_path, task_type='go-nogo')
            
            # Convert and analyze
            W, REC, spk, rs, all_fr, out, params = LIF_network_fnc(
                model_path, 50.0, stimulus, {'mode': 'none'}, 1, False
            )
            results[model_path] = {
                'spikes': np.sum(spk),
                'output': out[-1]
            }

Advanced Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spiking import format_spike_data
    import matplotlib.pyplot as plt
    
    # Load and convert model
    model_path = 'models/go-nogo/trained_model.mat'

    # Format spike data for analysis
    spike_data = format_spike_data(spk, params['dt'])

    # Print statistics
    print(f"Total spikes: {spike_data['total_spikes']}")
    print(f"Number of active neurons: {len(spike_data['active_neurons'])}")
    print(f"Mean firing rate: {np.mean(spike_data['firing_rates']):.2f} Hz")
    print(f"Spike rate: {spike_data['total_spikes'] / params['total_time']:.2f} spikes/s")

    # Plot firing rate distribution
    plt.figure(figsize=(8, 5))
    plt.hist(spike_data['firing_rates'], bins=30, alpha=0.7)
    plt.xlabel('Firing Rate (Hz)')
    plt.ylabel('Number of Neurons')
    plt.title('Firing Rate Distribution')
    plt.show()

Advanced Examples
----------------------------------------

Multi-Task Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare spiking network performance across different tasks:

.. code-block:: python

    tasks = ['go-nogo', 'xor', 'mante']
    model_paths = [
        'models/go-nogo/trained_model.mat',
        'models/xor/trained_model.mat', 
        'models/mante/trained_model.mat'
    ]

    for task, model_path in zip(tasks, model_paths):
        print(f"\nEvaluating {task} task...")
        
        # Optimize scaling factor
        lambda_grid_search(
            model_path=model_path,
            task_type=task,
            parallel=True
        )
        
        # Evaluate performance
        if task == 'go-nogo':
            eval_go_nogo(model_path=model_path, plot_results=True)

Parameter Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test how different LIF parameters affect conversion:

.. code-block:: python

    from spiking.utils import generate_lif_params

    model_path = 'models/go-nogo/trained_model.mat'
    scaling_factor = 50.0
    u = np.zeros((1, 201))
    u[0, 30:50] = 1

    # Test different time constants
    time_constants = [0.01, 0.02, 0.05, 0.1]
    
    for tm in time_constants:
        print(f"\nTesting membrane time constant: {tm}s")
        
        # This would require modifying LIF_network_fnc to accept custom parameters
        # or creating a custom implementation
        # Results would show how membrane dynamics affect spike timing 