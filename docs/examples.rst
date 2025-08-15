Examples
========

This section provides detailed examples of using the complete spikeRNN framework for training rate RNNs and converting them to spiking networks. 
The examples below can be run interactively in a Python environment or adapted into standalone scripts.

Complete Workflow Example
----------------------------------------

This example shows the complete workflow from training a rate RNN to converting it to a spiking network:

.. code-block:: python

    import torch
    import numpy as np
    from rate import FR_RNN_dale, set_gpu, create_default_config
    from rate.tasks import TaskFactory
    from spiking import LIF_network_fnc, lambda_grid_search, evaluate_task

    # Step 1: Set up GPU and configuration
    device = set_gpu('0', 0.3)
    config = create_default_config(N=200, P_inh=0.2, P_rec=0.2)

    # Step 2: Train rate RNN
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

    from rate.tasks import TaskFactory
    
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
        generate_input_stim_xor
        u, label = (settings)
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

Finding the optimal scaling factor is crucial for good performance.
You can run the grid search from the command line:

.. code-block:: bash

    python -m spiking.lambda_grid_search \
        --model_dir "models/go-nogo/P_rec_0.2_Taus_4.0_20.0" \
        --task_name go-nogo \
        --n_trials 100 \
        --scaling_factors 20:76:5


Or call the function from within a Python script:

.. code-block:: python

    from spiking import lambda_grid_search
    
    # Comprehensive grid search
    lambda_grid_search(
        model_dir='models/go-nogo/P_rec_0.2_Taus_4.0_20.0',
        task_name='go-nogo',
        n_trials=100,
        scaling_factors=(20, 76, 5)
    )

Task Performance Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also evaluate converted spiking networks directly from the command line. 
For example, to evaluate the Go-NoGo task for a specific model, run the following command from the spikeRNN directory:

.. code-block:: bash

    python -m spiking.eval_tasks --task go_nogo \
        --model_dir models/go-nogo/P_rec_0.2_Taus_4.0_20.0

If you have a specific scaling factor you want to use, you can specify it:

.. code-block:: bash

    python -m spiking.eval_tasks --task go_nogo \
        --model_dir models/go-nogo/P_rec_0.2_Taus_4.0_20.0 \
        --scaling_factor 50.0


Alternatively, you can call the evaluation function from a Python script:

.. code-block:: python

    from spiking.eval_tasks import evaluate_task
    
    # Evaluate Go-NoGo performance
    performance = evaluate_task(
        task_name='go_nogo',
        model_dir='models/go-nogo/P_rec_0.2_Taus_4.0_20.0'
    )

All registered tasks can be evaluated using the same interface:

.. code-block:: bash

    python -m spiking.eval_tasks --task xor --model_dir models/xor/
    python -m spiking.eval_tasks --task mante --model_dir models/mante/