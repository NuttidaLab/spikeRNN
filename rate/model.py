#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

# PyTorch adaptation of the continuous rate-based RNN 
# The original model is from the following paper:
# Kim, R., Hasson, D. V. Z. T., & Pehlevan, C. (2019). A framework for 
# reconciling rate and spike-based neuronal models. arXiv preprint arXiv:1904.05831.
# Original TensorFlow repository: https://github.com/rkim35/spikeRNN

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from typing import Dict, Any, Tuple, List, Union

'''
CONTINUOUS FIRING-RATE RNN CLASS
'''

class FR_RNN_dale(nn.Module):
    """
    Firing-rate RNN model for excitatory and inhibitory neurons
    Initialization of the firing-rate model with recurrent connections
    """
    def __init__(self, N: int, P_inh: float, P_rec: float, w_in: np.ndarray, som_N: int, 
                 w_dist: str, gain: float, apply_dale: bool, w_out: np.ndarray, device: torch.device) -> None:
        """
        Network initialization method.

        Args:
            N (int): Number of units (neurons).
            P_inh (float): Probability of a neuron being inhibitory.
            P_rec (float): Recurrent connection probability.
            w_in (np.ndarray): NxN weight matrix for the input stimuli.
            som_N (int): Number of SOM neurons (set to 0 for no SOM neurons).
            w_dist (str): Recurrent weight distribution ('gaus' or 'gamma').
            apply_dale (bool): Apply Dale's principle (True or False).
            w_out (np.ndarray): Nx1 readout weights.
            device (torch.device): PyTorch device.

        Note:
            Based on the probability (P_inh) provided above,
            the units in the network are classified into
            either excitatory or inhibitory. Next, the
            weight matrix is initialized based on the connectivity
            probability (P_rec) provided above.
        """
        super(FR_RNN_dale, self).__init__()
        
        self.N = N
        self.P_inh = P_inh
        self.P_rec = P_rec
        self.w_in = torch.tensor(w_in, dtype=torch.float32, device=device)
        self.som_N = som_N
        self.w_dist = w_dist
        self.gain = gain
        self.apply_dale = apply_dale
        self.device = device

        # Assign each unit as excitatory or inhibitory
        inh, exc, NI, NE, som_inh = self.assign_exc_inh()
        self.inh = inh
        self.som_inh = som_inh
        self.exc = exc
        self.NI = NI
        self.NE = NE

        # Initialize the weight matrix
        W, mask, som_mask = self.initialize_W()
        
        # Create learnable parameters
        self.w = nn.Parameter(torch.tensor(W, dtype=torch.float32, device=device))
        self.mask = torch.tensor(mask, dtype=torch.float32, device=device)
        self.som_mask = torch.tensor(som_mask, dtype=torch.float32, device=device)
        self.w_out = nn.Parameter(torch.tensor(w_out, dtype=torch.float32, device=device))
        self.b_out = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))

    def assign_exc_inh(self) -> Tuple[np.ndarray, np.ndarray, int, int, Union[np.ndarray, int]]:
        """
        Method to randomly assign units as excitatory or inhibitory (Dale's principle).

        Returns:
            Tuple[np.ndarray, np.ndarray, int, int, Union[np.ndarray, int]]: Tuple containing:
                - inh: Boolean array marking which units are inhibitory
                - exc: Boolean array marking which units are excitatory
                - NI: Number of inhibitory units
                - NE: Number of excitatory units
                - som_inh: Indices of "inh" for SOM neurons
        """
        # Apply Dale's principle
        if self.apply_dale == True:
            inh = np.random.rand(self.N, 1) < self.P_inh
            exc = ~inh
            NI = len(np.where(inh == True)[0])
            NE = self.N - NI

        # Do NOT apply Dale's principle
        else:
            inh = np.random.rand(self.N, 1) < 0 # no separate inhibitory units
            exc = ~inh
            NI = len(np.where(inh == True)[0])
            NE = self.N - NI

        if self.som_N > 0:
            som_inh = np.where(inh==True)[0][:self.som_N]
        else:
            som_inh = 0

        return inh, exc, NI, NE, som_inh

    def initialize_W(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method to generate and initialize the connectivity weight matrix, W.
        The weights are drawn from either gaussian or gamma distribution.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
                - w: NxN weights (all positive)
                - mask: NxN matrix of 1's (excitatory units) and -1's (for inhibitory units)
                - som_mask: NxN mask for SOM connectivity constraints

        Note:
            To compute the "full" weight matrix, simply
            multiply w and mask (i.e. w*mask)
        """
        # Weight matrix
        w = np.zeros((self.N, self.N), dtype = np.float32)
        idx = np.where(np.random.rand(self.N, self.N) < self.P_rec)
        if self.w_dist.lower() == 'gamma':
            w[idx[0], idx[1]] = np.random.gamma(2, 0.003, len(idx[0]))
        elif self.w_dist.lower() == 'gaus':
            w[idx[0], idx[1]] = np.random.normal(0, 1.0, len(idx[0]))
            w = w/np.sqrt(self.N*self.P_rec)*self.gain # scale by a gain to make it chaotic

        if self.apply_dale == True:
            w = np.abs(w)
        
        # Mask matrix
        mask = np.eye(self.N, dtype=np.float32)
        mask[np.where(self.inh==True)[0], np.where(self.inh==True)[0]] = -1

        # SOM mask matrix
        som_mask = np.ones((self.N, self.N), dtype=np.float32)
        if self.som_N > 0:
            for i in self.som_inh:
                som_mask[i, np.where(self.inh==True)[0]] = 0

        return w, mask, som_mask

    def load_net(self, model_dir: str) -> 'FR_RNN_dale':
        """
        Method to load pre-configured network settings.

        Args:
            model_dir (str): Path to the model directory containing saved parameters.

        Returns:
            FR_RNN_dale: The loaded network instance.
        """
        settings = scipy.io.loadmat(model_dir)
        self.N = settings['N'][0][0]
        self.som_N = settings['som_N'][0][0]
        self.inh = settings['inh']
        self.exc = settings['exc']
        self.inh = self.inh == 1
        self.exc = self.exc == 1
        self.NI = len(np.where(settings['inh'] == True)[0])
        self.NE = len(np.where(settings['exc'] == True)[0])
        
        # Update parameters
        self.mask = torch.tensor(settings['m'], dtype=torch.float32, device=self.device)
        self.som_mask = torch.tensor(settings['som_m'], dtype=torch.float32, device=self.device)
        self.w.data = torch.tensor(settings['w'], dtype=torch.float32, device=self.device)
        self.w_in = torch.tensor(settings['w_in'], dtype=torch.float32, device=self.device)
        self.b_out.data = torch.tensor(settings['b_out'], dtype=torch.float32, device=self.device)
        self.w_out.data = torch.tensor(settings['w_out'], dtype=torch.float32, device=self.device)

        return self
    
    def display(self) -> None:
        """
        Method to print the network setup.
        """
        print('Network Settings')
        print('====================================')
        print('Number of Units: ', self.N)
        print('\t Number of Excitatory Units: ', self.NE)
        print('\t Number of Inhibitory Units: ', self.NI)
        print('Weight Matrix, W')
        full_w = (self.w * self.mask).cpu().numpy()
        zero_w = len(np.where(full_w == 0)[0])
        pos_w = len(np.where(full_w > 0)[0])
        neg_w = len(np.where(full_w < 0)[0])
        print('\t Zero Weights: %2.2f %%' % (zero_w/(self.N*self.N)*100))
        print('\t Positive Weights: %2.2f %%' % (pos_w/(self.N*self.N)*100))
        print('\t Negative Weights: %2.2f %%' % (neg_w/(self.N*self.N)*100))

    def forward(self, stim: torch.Tensor, taus: List[float], training_params: Dict[str, Any], 
                settings: Dict[str, Any]) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], 
                List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
                torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the RNN.

        Args:
            stim (torch.Tensor): Input stimulus tensor of shape (input_dim, T).
            taus (List[float]): Time constants (either single value or [min, max] range).
            training_params (Dict[str, Any]): Training parameters including activation function.
            settings (Dict[str, Any]): Task settings including T (trial duration) and DeltaT (sampling rate).

        Returns:
            Tuple containing multiple tensors:
            
            - stim: Input stimulus tensor
            - x: List of synaptic current tensors over time
            - r: List of firing rate tensors over time  
            - o: List of output tensors over time
            - w: Recurrent weight matrix
            - w_in: Input weight matrix
            - mask: Dale's principle mask matrix
            - som_mask: SOM connectivity mask matrix
            - w_out: Output weight matrix
            - b_out: Output bias
            - taus_gaus: Time constant parameters
        """
        T = settings['T']
        DeltaT = settings['DeltaT']
        
        # Initialize taus_gaus for time constants
        if len(taus) > 1:
            taus_gaus = torch.randn(self.N, 1, device=self.device, requires_grad=True)
        else:
            taus_gaus = torch.randn(self.N, 1, device=self.device, requires_grad=False)
        
        # Synaptic currents and firing-rates
        x = []
        r = []
        x.append(torch.randn(self.N, 1, device=self.device) / 100)
        
        # Initial firing rate with activation function
        if training_params['activation'] == 'sigmoid':
            r.append(torch.sigmoid(x[0]))
        elif training_params['activation'] == 'clipped_relu':
            r.append(torch.clamp(F.relu(x[0]), 0, 20))
        elif training_params['activation'] == 'softplus':
            r.append(torch.clamp(F.softplus(x[0]), 0, 20))
        
        # Output list
        o = []
        
        # Forward pass through time
        for t in range(1, T):
            if self.apply_dale == True:
                # Parametrize the weight matrix to enforce exc/inh synaptic currents
                w_pos = F.relu(self.w)
            else:
                w_pos = self.w
            
            # Compute effective weight matrix
            ww = torch.matmul(w_pos, self.mask)
            ww = ww * self.som_mask
            
            # Compute time constants
            if len(taus) > 1:
                taus_sig = torch.sigmoid(taus_gaus) * (taus[1] - taus[0]) + taus[0]
            else:
                taus_sig = taus[0]
            
            # Update synaptic currents
            next_x = (1 - DeltaT / taus_sig) * x[t-1] + \
                    (DeltaT / taus_sig) * (torch.matmul(ww, r[t-1]) + \
                    torch.matmul(self.w_in, stim[:, t-1:t])) + \
                    torch.randn(self.N, 1, device=self.device) / 10
            
            x.append(next_x)
            
            # Apply activation function
            if training_params['activation'] == 'sigmoid':
                r.append(torch.sigmoid(next_x))
            elif training_params['activation'] == 'clipped_relu':
                r.append(torch.clamp(F.relu(next_x), 0, 20))
            elif training_params['activation'] == 'softplus':
                r.append(torch.clamp(F.softplus(next_x), 0, 20))
            
            # Compute output
            next_o = torch.matmul(self.w_out, r[t]) + self.b_out
            o.append(next_o)
        
        return stim, x, r, o, self.w, self.w_in, self.mask, self.som_mask, self.w_out, self.b_out, taus_gaus

    def lesion_w(self, lesion_percentage: float) -> None:
        """
        Applies lesions to the recurrent weight matrix by setting a
        percentage of connections to zero.

        Args:
            lesion_percentage (float): The percentage of connections to remove (0.0 to 1.0).
        """
        with torch.no_grad():
            # Clone the original weight tensor to avoid modifying it directly during iteration
            w_lesioned = self.w.clone()

            # Get the indices of non-diagonal elements that are non-zero
            non_diagonal_indices = torch.nonzero(self.w, as_tuple=False)

            # Calculate the number of connections to remove
            num_to_lesion = int(lesion_percentage * len(non_diagonal_indices))

            # Randomly select connections to lesion
            lesion_indices = torch.randperm(len(non_diagonal_indices))[:num_to_lesion]
            indices_to_zero = non_diagonal_indices[lesion_indices]

            # Set the selected weights to zero
            for index in indices_to_zero:
                w_lesioned[index[0], index[1]] = 0

            # Update the model's weight parameter
            self.w.data = w_lesioned
    
    def lesion_w_by_type(self, lesion_percentage: float) -> None:
        """
        Applies lesions by setting an equal percentage of existing connections
        to zero for each neuron type pairing (E-E, E-I, I-E, I-I).

        Args:
            lesion_percentage (float): The percentage of connections to remove (0.0 to 1.0).
        """
        if not 0.0 <= lesion_percentage <= 1.0:
            raise ValueError("lesion_percentage must be between 0.0 and 1.0")

        with torch.no_grad():
            w_lesioned = self.w.clone()

            # Convert boolean numpy arrays to tensors for masking
            exc_mask_t = torch.from_numpy(self.exc.flatten()).bool().to(self.w.device)
            inh_mask_t = torch.from_numpy(self.inh.flatten()).bool().to(self.w.device)

            # Define the four connection types using broadcasting to create masks
            # Note: W[i, j] is from neuron j (column) to neuron i (row)
            # mask[i, j] is True if it's a connection from type j to type i
            connection_masks = {
                "E_E": torch.outer(exc_mask_t, exc_mask_t), # To E (rows), From E (cols)
                "E_I": torch.outer(inh_mask_t, exc_mask_t), # To I (rows), From E (cols)
                "I_E": torch.outer(exc_mask_t, inh_mask_t), # To E (rows), From I (cols)
                "I_I": torch.outer(inh_mask_t, inh_mask_t)  # To I (rows), From I (cols)
            }
            # Lesion each connection type separately
            for conn_type, mask in connection_masks.items():
                # Find the indices of existing (non-zero) connections for this type
                indices = torch.nonzero(w_lesioned * mask, as_tuple=False)

                if len(indices) == 0:
                    continue  # No connections of this type to lesion

                # Calculate the number of connections to remove
                num_to_lesion = int(lesion_percentage * len(indices))

                if num_to_lesion > 0:
                    # Randomly select connections to lesion
                    lesion_indices_perm = torch.randperm(len(indices))[:num_to_lesion]
                    indices_to_zero = indices[lesion_indices_perm]

                    # Set the selected weights to zero
                    w_lesioned[indices_to_zero[:, 0], indices_to_zero[:, 1]] = 0

            # Update the model's weight parameter
            self.w.data = w_lesioned


'''
Task-specific input signals
'''
def generate_input_stim_go_nogo(settings: Dict[str, Any], seed: bool = False) -> Tuple[np.ndarray, int]:
    """
    Generate the input stimulus matrix for the Go-NoGo task.

    Args:
        settings (Dict[str, Any]): Dictionary containing the following keys:
            - T: Duration of a single trial (in steps)
            - stim_on: Stimulus starting time (in steps)
            - stim_dur: Stimulus duration (in steps)
            - taus: Time-constants (in steps)
            - DeltaT: Sampling rate

    Returns:
        Tuple[np.ndarray, int]: Tuple containing:
            - u: 1xT stimulus matrix
            - label: Either 1 (Go trial) or 0 (NoGo trial)
    """
    if seed == True:
        np.random.seed(42)
    
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    u = np.zeros((1, T)) #+ np.random.randn(1, T)
    u_lab = np.zeros((2, 1))
    if np.random.rand() <= 0.50:
        u[0, stim_on:stim_on+stim_dur] = 1
        label = 1
    else:
        label = 0 

    return u, label

def generate_input_stim_xor(settings: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    """
    Generate the input stimulus matrix for the XOR task.

    Args:
        settings (Dict[str, Any]): Dictionary containing the following keys:
            - T: Duration of a single trial (in steps)
            - stim_on: Stimulus starting time (in steps)
            - stim_dur: Stimulus duration (in steps)
            - delay: Delay between two stimuli (in steps)
            - taus: Time-constants (in steps)
            - DeltaT: Sampling rate

    Returns:
        Tuple[np.ndarray, str]: Tuple containing:
            - u: 2xT stimulus matrix
            - label: Either 'same' or 'diff'
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    delay = settings['delay']

    # Initialize u
    u = np.zeros((2, T))

    # XOR task
    labs = []
    if np.random.rand() < 0.50:
        u[0, stim_on:stim_on+stim_dur] = 1
        labs.append(1)
    else:
        u[0, stim_on:stim_on+stim_dur] = -1
        labs.append(-1)

    if np.random.rand() < 0.50:
        u[1, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay] = 1
        labs.append(1)
    else:
        u[1, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay] = -1
        labs.append(-1)

    if np.prod(labs) == 1:
        label = 'same'
    else:
        label = 'diff'

    return u, label

def generate_input_stim_mante(settings: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    """
    Generate the input stimulus matrix for the sensory integration task from Mante et al (2013).

    Args:
        settings (Dict[str, Any]): Dictionary containing the following keys:
            - T: Duration of a single trial (in steps)
            - stim_on: Stimulus starting time (in steps)
            - stim_dur: Stimulus duration (in steps)
            - DeltaT: Sampling rate

    Returns:
        Tuple[np.ndarray, int]: Tuple containing:
            - u: 4xT stimulus matrix
            - label: Either +1 or -1 (the correct decision)
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    # Initialize stimulus
    u = np.zeros((4, T))

    # Color task
    color_input = 2.5*(np.random.rand()-0.5) # [-1.25, 1.25]
    motion_input = 2.5*(np.random.rand()-0.5) # [-1.25, 1.25]

    # Context signal
    if np.random.rand() < 0.50:
        # Context = color task
        u[0, stim_on:stim_on+stim_dur] = 1 # context cue
        u[1, stim_on:stim_on+stim_dur] = color_input # color input
        u[2, stim_on:stim_on+stim_dur] = motion_input # motion input (irrelevant in this context)

        if color_input > 0:
            label = 1 # choose option 1
        else:
            label = -1 # choose option 2

    else:
        # Context = motion task
        u[0, stim_on:stim_on+stim_dur] = -1 # context cue
        u[1, stim_on:stim_on+stim_dur] = color_input # color input (irrelevant in this context)
        u[2, stim_on:stim_on+stim_dur] = motion_input # motion input

        if motion_input > 0:
            label = 1 # choose option 1
        else:
            label = -1 # choose option 2

    return u, label

def generate_target_continuous_go_nogo(settings: Dict[str, Any], label: int, seed: bool = False) -> np.ndarray:
    """
    Generate the target output signal for the Go-NoGo task.

    Args:
        settings (Dict[str, Any]): Dictionary containing the following keys:
            - T: Duration of a single trial (in steps)
            - stim_on: Stimulus starting time (in steps)
            - stim_dur: Stimulus duration (in steps)
        label (int): Either 1 (Go trial) or 0 (NoGo trial).

    Returns:
        np.ndarray: 1xT target signal array.
    """
    if seed == True:
        np.random.seed(42)
    
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    target = np.zeros((T-1,))
    resp_onset = stim_on + stim_dur
    if label == 1:
        target[resp_onset:] = 1
    else:
        target[resp_onset:] = 0

    return target

def generate_target_continuous_xor(settings: Dict[str, Any], label: str) -> np.ndarray:
    """
    Generate the target output signal for the XOR task.

    Args:
        settings (Dict[str, Any]): Dictionary containing task parameters.
        label (str): Either 'same' or 'diff'.

    Returns:
        np.ndarray: A 1D target signal array of shape (T,).
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    delay = settings['delay']

    # Calculate the time when the second stimulus presentation ends
    task_end_T = stim_on + (2 * stim_dur) + delay

    # Initialize the target signal array with shape (1, T)
    z = np.zeros((1, T))

    # Define the target window: starts 10 steps after the task ends and lasts for 100 steps
    target_onset = 10 + task_end_T
    target_offset = target_onset + 100

    # Assign the target value based on the label
    if label == 'same':
        z[0, target_onset:target_offset] = 1
    elif label == 'diff':
        z[0, target_onset:target_offset] = -1

    return np.squeeze(z)


def generate_target_continuous_mante(settings: Dict[str, Any], label: int) -> np.ndarray:
    """
    Generate the target output signal for the sensory integration task.

    Args:
        settings (Dict[str, Any]): Dictionary containing task parameters.
        label (int): Either +1 or -1, the correct decision.

    Returns:
        np.ndarray: A 1D target signal array of shape (T,).
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    # Initialize the target signal array with shape (1, T)
    z = np.zeros((1, T))
    
    # Calculate the target onset time dynamically
    target_onset = stim_on + stim_dur

    # Assign the target value from the onset time to the end of the trial
    if label == 1:
        z[0, target_onset:] = 1
    else:
        z[0, target_onset:] = -1

    # Squeeze the array to shape (T,) to match the original's output
    return np.squeeze(z)

def loss_op(o: List[torch.Tensor], z: np.ndarray, training_params: Dict[str, Any]) -> torch.Tensor:
    """
    Define loss function for training.

    Args:
        o (List[torch.Tensor]): List of output values from the network.
        z (np.ndarray): Target values.
        training_params (Dict[str, Any]): Dictionary containing training parameters 
                                         including 'loss_fn' key.

    Returns:
        torch.Tensor: Loss function value.
    """
    # Loss function
    loss = torch.tensor(0.0, requires_grad=True)
    loss_fn = training_params['loss_fn']
    
    z_tensor = torch.tensor(z, dtype=torch.float32, device=o[0].device)
    
    for i in range(len(o)):
        if loss_fn.lower() == 'l1': # mean absolute error (MAE)
            loss = loss + torch.norm(o[i].squeeze() - z_tensor[i], p=1)
        elif loss_fn.lower() == 'l2': # root mean squared error (RMSE)
            loss = loss + (o[i].squeeze() - z_tensor[i])**2
    
    if loss_fn.lower() == 'l2':
        loss = torch.sqrt(loss)

    return loss

def eval_rnn(net: FR_RNN_dale, settings: Dict[str, Any], u: np.ndarray, 
             device: torch.device) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
    """
    Evaluate a trained PyTorch RNN.

    Args:
        net (FR_RNN_dale): Trained FR_RNN_dale model.
        settings (Dict[str, Any]): Dictionary containing task settings.
        u (np.ndarray): Stimulus matrix.
        device (torch.device): PyTorch device.

    Returns:
        Tuple[List[float], List[np.ndarray], List[np.ndarray]]: Tuple containing:
            - o: Output vector (list of floats)
            - r: Firing rates (list of numpy arrays)
            - x: Synaptic currents (list of numpy arrays)
    """
    T = settings['T']
    DeltaT = settings['DeltaT']
    taus = settings['taus']
    
    net.eval()
    with torch.no_grad():
        # Convert input to tensor
        u_tensor = torch.tensor(u, dtype=torch.float32, device=device)
        
        # Initialize
        x = []
        r = []
        x.append(torch.randn(net.N, 1, device=device) / 100)
        r.append(torch.sigmoid(x[0]))  # Default to sigmoid
        
        o = []
        
        for t in range(1, T):
            if net.apply_dale:
                w_pos = F.relu(net.w)
            else:
                w_pos = net.w
            
            ww = torch.matmul(w_pos, net.mask)
            ww = ww * net.som_mask
            
            if len(taus) > 1:
                taus_sig = taus[0]  # Use first tau for evaluation
            else:
                taus_sig = taus[0]
            
            next_x = (1 - DeltaT / taus_sig) * x[t-1] + \
                    (DeltaT / taus_sig) * (torch.matmul(ww, r[t-1]) + \
                    torch.matmul(net.w_in, u_tensor[:, t-1:t])) + \
                    torch.randn(net.N, 1, device=device) / 10
            
            x.append(next_x)
            r.append(torch.sigmoid(next_x))  # Default to sigmoid
            
            next_o = torch.matmul(net.w_out, r[t]) + net.b_out
            o.append(next_o.item())
    
    return o, [r_t.cpu().numpy() for r_t in r], [x_t.cpu().numpy() for x_t in x] 