#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Adaptation from Kim R., Li Y., & Sejnowski TJ. Simple Framework for Constructing Functional Spiking 
# Recurrent Neural Networks. Preprint at BioRxiv https://www.biorxiv.org/content/10.1101/579706v2 (2019).
# Original TensorFlow repository: https://github.com/rkim35/spikeRNN

import os, sys
import time
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import datetime

# Import utility functions
from utils import set_gpu
from utils import restricted_float
from utils import str2bool

# Import the continuous rate model
from model import FR_RNN_dale

# Import the tasks
from tasks import TaskFactory

from model import loss_op

# Parse input arguments
parser = argparse.ArgumentParser(description='Training rate RNNs')
parser.add_argument('--gpu', required=False,
        default='0', help="Which gpu to use")
parser.add_argument("--gpu_frac", required=False,
        type=restricted_float, default=0.4,
        help="Fraction of GPU mem to use")
parser.add_argument("--n_trials", required=True,
        type=int, default=200, help="Number of epochs")
parser.add_argument("--mode", required=True,
        type=str, default='Train', help="Train or Eval")
parser.add_argument("--output_dir", required=True,
        type=str, help="Model output path")
parser.add_argument("--N", required=True,
        type=int, help="Number of neurons")
parser.add_argument("--gain", required=False,
        type=float, default = 1.5, help="Gain for the connectivity weight initialization")
parser.add_argument("--P_inh", required=False,
        type=restricted_float, default = 0.20,
        help="Proportion of inhibitory neurons")
parser.add_argument("--P_rec", required=False,
        type=restricted_float, default = 0.20,
        help="Connectivity probability")
parser.add_argument("--som_N", required=True,
        type=int, default = 0, help="Number of SST neurons")
parser.add_argument("--task", required=True,
        type=str, help="Task (XOR, sine, etc...)")
parser.add_argument("--act", required=True,
        type=str, default='sigmoid', help="Activation function (sigmoid, clipped_relu)")
parser.add_argument("--loss_fn", required=True,
        type=str, default='l2', help="Loss function (either L1 or L2)")
parser.add_argument("--apply_dale", required=True,
        type=str2bool, default='True', help="Apply Dale's principle?")
parser.add_argument("--decay_taus", required=True,
        nargs='+', type=float,
        help="Synaptic decay time-constants (in time-steps). If only one number is given, then all\
        time-constants set to that value (i.e. not trainable). Otherwise specify two numbers (min, max).")
args = parser.parse_args()

# Set up device
device = set_gpu(args.gpu, args.gpu_frac)
print(f'Using device: {device}')

# Set up the output dir where the output model will be saved
out_dir = os.path.join(args.output_dir, 'models', args.task.lower())
if args.apply_dale == False:
    out_dir = os.path.join(out_dir, 'NoDale')
if len(args.decay_taus) > 1:
    out_dir = os.path.join(out_dir, 'P_rec_' + str(args.P_rec) + '_Taus_' + str(args.decay_taus[0]) + '_' + str(args.decay_taus[1]))
else:
    out_dir = os.path.join(out_dir, 'P_rec_' + str(args.P_rec) + '_Tau_' + str(args.decay_taus[0]))

if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

# Number of units/neurons
N = args.N
som_N = args.som_N; # number of SST neurons 

# Define task-specific parameters
# NOTE: Each time step is 5 ms
if args.task.lower() == 'go-nogo':
    # GO-NoGo task
    settings = {
            'T': 200, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 25, # input stim duration (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name
            }
elif args.task.lower() == 'xor':
    # XOR task 
    settings = {
            'T': 300, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 50, # input stim duration (in steps)
            'delay': 10, # delay b/w the two stimuli (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name
            }
elif args.task.lower() == 'mante':
    # Sensory integration task
    settings = {
            'T': 500, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 200, # input stim duration (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name
            }

'''
Initialize the input and output weight matrices
'''
# Go-Nogo task
if args.task.lower() == 'go-nogo':
    w_in = np.float32(np.random.randn(N, 1))
    w_out = np.float32(np.random.randn(1, N)/100)

# XOR task
elif args.task.lower() == 'xor':
    w_in = np.float32(np.random.randn(N, 2))
    w_out = np.float32(np.random.randn(1, N)/100)

# Sensory integration task
elif args.task.lower() == 'mante':
    w_in = np.float32(np.random.randn(N, 4))
    w_out = np.float32(np.random.randn(1, N)/100)

'''
Initialize the continuous rate model
'''
P_inh = args.P_inh # inhibitory neuron proportion
P_rec = args.P_rec # initial connectivity probability (i.e. sparsity degree)
print('P_rec set to ' + str(P_rec))

w_dist = 'gaus' # recurrent weight distribution (Gaussian or Gamma)
net = FR_RNN_dale(N, P_inh, P_rec, w_in, som_N, w_dist, args.gain, args.apply_dale, w_out, device)
print('Intialized the network...')

# Snapshot initial weights before any training
w0 = net.w.clone().detach()

'''
Define the training parameters (learning rate, training termination criteria, etc...)
'''
training_params = {
        'learning_rate': 0.01, # learning rate
        'loss_threshold': 7, # loss threshold (when to stop training)
        'eval_freq': 100, # how often to evaluate task perf
        'eval_tr': 100, # number of trials for eval
        'eval_amp_threh': 0.3, # amplitude threshold during response window
        'activation': args.act.lower(), # activation function
        'loss_fn': args.loss_fn.lower(), # loss function ('L1' or 'L2')
        'P_rec': 0.20
        }

'''
Set up optimizer
'''
if args.mode.lower() == 'train':
    optimizer = optim.Adam(net.parameters(), lr=training_params['learning_rate'])
    print('Set up optimizer...')

'''
Start training
'''
if args.mode.lower() == 'train':
    print('Training started...')
    training_success = False

    # Get initial states
    task = TaskFactory.create_task(args.task.lower().replace('-', '_'), settings)
    u, target, label = task.simulate_trial()

    # Convert to tensor
    u_tensor = torch.tensor(u, dtype=torch.float32, device=device)
    
    # Get initial network outputs
    stim0, x0, r0, o0, _, w_in0, m0, som_m0, w_out0, b_out0, taus_gaus0 = \
            net.forward(u_tensor, settings['taus'], training_params, settings)

    # For storing all the loss vals
    losses = np.zeros((args.n_trials,))

    for tr in range(args.n_trials):
        start_time = time.time()
        
        # Zero gradients
        optimizer.zero_grad()

        # Generate a task-specific input signal
        u, target, label = task.simulate_trial()

        print("Trial " + str(tr) + ': ' + str(label))

        # Convert to tensor
        u_tensor = torch.tensor(u, dtype=torch.float32, device=device)

        # Forward pass
        stim, x, r, o, w, w_in, m, som_m, w_out, b_out, taus_gaus = \
                net.forward(u_tensor, settings['taus'], training_params, settings)

        # Compute loss
        t_loss = loss_op(o, target, training_params)

        # Backward pass
        t_loss.backward()
        optimizer.step()

        print('Loss: ', t_loss.item())
        losses[tr] = t_loss.item()

        '''
        Evaluate the model and determine if the training termination criteria are met
        '''
        # Go-NoGo task
        if args.task.lower() == 'go-nogo':
            resp_onset = settings['stim_on'] + settings['stim_dur']
            if (tr-1)%training_params['eval_freq'] == 0:
                eval_perf = np.zeros((1, training_params['eval_tr']))
                eval_losses = np.zeros((1, training_params['eval_tr']))
                eval_os = np.zeros((training_params['eval_tr'], settings['T']-1))
                eval_labels = np.zeros((training_params['eval_tr'], ))
                
                net.eval()
                with torch.no_grad():
                    for ii in range(eval_perf.shape[-1]):
                        eval_u, eval_target, eval_label = task.simulate_trial()
                        eval_u_tensor = torch.tensor(eval_u, dtype=torch.float32, device=device)
                        _, _, _, eval_o, _, _, _, _, _, _, _ = \
                                net.forward(eval_u_tensor, settings['taus'], training_params, settings)
                        
                        # Convert outputs to numpy
                        eval_o_np = torch.cat(eval_o).cpu().numpy().flatten()
                        eval_l = loss_op(eval_o, eval_target, training_params).item()
                        
                        eval_losses[0, ii] = eval_l
                        eval_os[ii, :] = eval_o_np
                        eval_labels[ii, ] = eval_label
                        if eval_label == 1:
                            if np.max(eval_o_np[resp_onset:]) > training_params['eval_amp_threh']:
                                eval_perf[0, ii] = 1
                        else:
                            if np.max(np.abs(eval_o_np[resp_onset:])) < 0.3:
                                eval_perf[0, ii] = 1
                
                net.train()
                eval_perf_mean = np.nanmean(eval_perf, 1)[0]
                eval_loss_mean = np.nanmean(eval_losses, 1)[0]
                print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean, eval_loss_mean))

                if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.95 and tr > 1500:
                    # For this task, the minimum number of trials required is set to 1500 to 
                    # ensure that the trained rate model is stable.
                    training_success = True
                    break

        # XOR task
        elif args.task.lower() == 'xor':
            if (tr-1)%training_params['eval_freq'] == 0:
                eval_perf = np.zeros((1, training_params['eval_tr']))
                eval_losses = np.zeros((1, training_params['eval_tr']))
                eval_os = np.zeros((training_params['eval_tr'], settings['T']-1))
                eval_labels = []
                
                # Calculate evaluation window dynamically
                stim_on = settings['stim_on']
                stim_dur = settings['stim_dur']
                delay = settings['delay']
                task_end_T = stim_on + (2 * stim_dur) + delay
                eval_onset = 10 + task_end_T
                eval_offset = eval_onset + 100
                
                net.eval()
                with torch.no_grad():
                    for ii in range(eval_perf.shape[-1]):
                        eval_u, eval_target, eval_label = task.simulate_trial()
                        eval_u_tensor = torch.tensor(eval_u, dtype=torch.float32, device=device)
                        _, _, _, eval_o, _, _, _, _, _, _, _ = \
                                net.forward(eval_u_tensor, settings['taus'], training_params, settings)
                        
                        # Convert outputs to numpy
                        eval_o_np = torch.cat(eval_o).cpu().numpy().flatten()
                        eval_l = loss_op(eval_o, eval_target, training_params).item()
                        
                        eval_losses[0, ii] = eval_l
                        eval_os[ii, :] = eval_o_np
                        eval_labels.append(eval_label)
                        
                        if eval_label == 'same':
                            if np.max(eval_o_np[eval_onset:eval_offset]) > training_params['eval_amp_threh']:
                                eval_perf[0, ii] = 1
                        else:
                            if np.min(eval_o_np[eval_onset:eval_offset]) < -training_params['eval_amp_threh']:
                                eval_perf[0, ii] = 1
                
                net.train()
                eval_perf_mean = np.nanmean(eval_perf, 1)[0]
                eval_loss_mean = np.nanmean(eval_losses, 1)[0]
                print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean, eval_loss_mean))

                if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.95:
                    training_success = True
                    break

        # Sensory integration task
        elif args.task.lower() == 'mante':
            resp_onset = settings['stim_on'] + settings['stim_dur']
            if (tr-1)%training_params['eval_freq'] == 0:
                eval_perf = np.zeros((1, training_params['eval_tr']))
                eval_losses = np.zeros((1, training_params['eval_tr']))
                eval_os = np.zeros((training_params['eval_tr'], settings['T']-1))
                eval_labels = np.zeros((training_params['eval_tr'], ))
                
                net.eval()
                with torch.no_grad():
                    for ii in range(eval_perf.shape[-1]):
                        eval_u, eval_target, eval_label = task.simulate_trial()
                        eval_u_tensor = torch.tensor(eval_u, dtype=torch.float32, device=device)
                        _, _, _, eval_o, _, _, _, _, _, _, _ = \
                                net.forward(eval_u_tensor, settings['taus'], training_params, settings)
                        
                        # Convert outputs to numpy
                        eval_o_np = torch.cat(eval_o).cpu().numpy().flatten()
                        eval_l = loss_op(eval_o, eval_target, training_params).item()
                        
                        eval_losses[0, ii] = eval_l
                        eval_os[ii, :] = eval_o_np
                        eval_labels[ii, ] = eval_label
                        if eval_label == 1:
                            if np.max(eval_o_np[resp_onset:]) > training_params['eval_amp_threh']:
                                eval_perf[0, ii] = 1
                        else:
                            if np.min(eval_o_np[resp_onset:]) < -training_params['eval_amp_threh']:
                                eval_perf[0, ii] = 1
                
                net.train()
                eval_perf_mean = np.nanmean(eval_perf, 1)[0]
                eval_loss_mean = np.nanmean(eval_losses, 1)[0]
                print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean, eval_loss_mean))

                if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.95:
                    training_success = True
                    break

    elapsed_time = time.time() - start_time
    # print(elapsed_time)

    # Save the trained params in a .mat file
    w = net.w.clone().detach() # Snapshot final weights after training
    var = {}
    var['x0'] = x0[0].detach().cpu().numpy()
    var['r0'] = r0[0].detach().cpu().numpy()
    var['w0'] = w0.detach().cpu().numpy()
    var['taus_gaus0'] = taus_gaus0.detach().cpu().numpy()
    var['w_in0'] = w_in0.detach().cpu().numpy()
    var['u'] = u
    var['o'] = torch.cat(o).detach().cpu().numpy().flatten()
    var['w'] = w.detach().cpu().numpy()
    var['x'] = x[-1].detach().cpu().numpy()
    var['target'] = target
    var['w_out'] = w_out.detach().cpu().numpy()
    var['r'] = r[-1].detach().cpu().numpy()
    var['m'] = m.detach().cpu().numpy()
    var['som_m'] = som_m.detach().cpu().numpy()
    var['N'] = N
    var['exc'] = net.exc
    var['inh'] = net.inh
    var['w_in'] = w_in.detach().cpu().numpy()
    var['b_out'] = b_out.detach().cpu().numpy()
    var['som_N'] = som_N
    var['losses'] = losses
    var['taus'] = settings['taus']
    var['eval_perf_mean'] = eval_perf_mean
    var['eval_loss_mean'] = eval_loss_mean
    var['eval_os'] = eval_os
    var['eval_labels'] = eval_labels
    var['taus_gaus'] = taus_gaus.detach().cpu().numpy()
    var['tr'] = tr
    var['activation'] = training_params['activation']
    fname_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    if len(settings['taus']) > 1:
        fname = 'Task_{}_N_{}_Taus_{}_{}_Act_{}_{}.mat'.format(args.task.lower(), N, settings['taus'][0], 
                settings['taus'][1], training_params['activation'], fname_time)
    elif len(settings['taus']) == 1:
        fname = 'Task_{}_N_{}_Tau_{}_Act_{}_{}.mat'.format(args.task.lower(), N, settings['taus'][0], 
                training_params['activation'], fname_time)
    # Convert to float64
    var = {k: (v.astype(np.float64) if isinstance(v, np.ndarray) else v) for k, v in var.items()}
    scipy.io.savemat(os.path.join(out_dir, fname), var)
    
    # Also save the PyTorch model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'settings': settings,
        'training_params': training_params,
        'losses': losses,
        'final_loss': t_loss.item(),
        'trial': tr,
        }, os.path.join(out_dir, fname.replace('.mat', '.pth'))) 
    
    
    
    # Example command to run the training
    """
    python main.py --gpu 0 --gpu_frac 0.20 \
        --n_trials 5000 --mode train \
        --N 200 --P_inh 0.20 --som_N 0 --apply_dale True \
        --gain 1.5 --task go-nogo --act sigmoid --loss_fn l2 \
        --decay_taus 4 20 --output_dir ../
    """