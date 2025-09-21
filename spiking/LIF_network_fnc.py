"""
Function for converting a trained rate RNN to a spiking RNN (leaky integrate-and-fire).

The function:

* Converts a trained rate RNN to a spiking RNN (leaky integrate-and-fire)
* Uses the LIF model for the spiking RNN
"""

# PyTorch adaptation of the function to perform the one-to-one mapping
# from a trained rate RNN to a spiking RNN (leaky integrate-and-fire).

# The original model is from the following paper:
# Kim, R., Hasson, D. V. Z. T., & Pehlevan, C. (2019). A framework for 
# reconciling rate and spike-based neuronal models. arXiv preprint arXiv:1904.05831.
# Original repository: https://github.com/rkim35/spikeRNN

# NOTE: LIF network implementation modified from LIFFORCESINE.m from NicolaClopath2017 
# (https://senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=190565&file=/NicolaClopath2017/)

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import os
import warnings
warnings.filterwarnings("ignore")

def LIF_network_fnc(model_or_path, scaling_factor, u, stims, downsample, use_initial_weights,
                    lesion_percentage: float = 0.0,
                    lesion_scale: float = 0.0,
                    lesion_by_type: bool = False,
                    exclude_self_connections: bool = True,
                    seed: int = None):
    """
    Convert a trained rate RNN to a spiking RNN (leaky integrate-and-fire).
    
    Args:
      - model_or_path: trained model full path (directory + filename) or loaded model data (dict)
      - scaling_factor: scaling factor for transferring weights from rate to spk
      - u: input stimulus to be used
      - stims: dict for artificial stimulations (to model optogenetic stim)
          - mode: "none", "exc" (depolarizing), or "inh" (hyperpolarizing)
          - dur: [stim_onset stim_offset]
          - units: vector containing unit indices to be stimulated
      - downsample: downsample factor (1 => no downsampling, 2 => every other sample, etc...)
                    While downsample > 1 can speed up the conversion, the LIF network
                    might not be as robust as the one without downsampling
      - use_initial_weights: whether to use w0 (random initial weights). This is mainly used for testing.
      - lesion_percentage: fraction (0..1) of existing connections to attenuate
      - lesion_scale: multiplicative factor applied to selected weights (0..1)
      - lesion_by_type: if True, lesion equally within E-E, E-I, I-E, I-I
      - exclude_self_connections: if True, do not lesion self-connections (i==j)
      - seed: int for reproducible random selection

    Returns:
      - W: recurrent connectivity matrix scaled by the scaling factor (N x N)
      - REC: membrane voltage from all the units (N x t)
      - spk: binary matrix indicating spikes (N x t)
      - rs: firing rates from all the units (N x t)
      - all_fr: average firing rates from all the units (N x 1)
      - out: network output (1 x t)
      - params: dict containing sampling rate info
    """

    #------------------------------------------------------
    # Extract the number of units and the connectivity
    # matrix from the trained continuous rate model
    #------------------------------------------------------
    if isinstance(model_or_path, str):
        model_data = sio.loadmat(model_or_path)
    else:
        model_data = model_or_path
    
    # Convert all numpy arrays to float64
    for k in model_data:
        if isinstance(model_data[k], np.ndarray):
            try:
                model_data[k] = model_data[k].astype(np.float64)
            except ValueError:
                # Not a numeric array, skip conversion
                pass
              
    w_in = model_data['w_in']
    w = model_data['w']
    w0 = model_data['w0']
    N = int(model_data['N'].item())
    m = model_data['m']
    som_m = model_data['som_m']
    w_out = model_data['w_out']
    
    # If a weight vector was loaded as 1D, reshape it to a 2D column/row vector
    if w_in.ndim == 1:
        w_in = w_in.reshape(N, 1)
    if w_out.ndim == 1:
        w_out = w_out.reshape(1, N)

    inh = model_data['inh'].flatten()
    exc = model_data['exc'].flatten()
    taus_gaus0 = model_data['taus_gaus0']
    taus_gaus = model_data['taus_gaus']
    taus = model_data['taus']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if use_initial_weights:
        w = np.matmul(w0, m) * som_m
    else:
        w = np.matmul(w, m) * som_m

    # Scale the connectivity weights by the optimal scaling factor 
    W = torch.tensor(w / scaling_factor, dtype=torch.float64, device=device)
    
    # Lesion helpers
    def _rng(device, seed):
        if seed is None:
            return torch.Generator(device=device)
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return g

    def _apply_lesion_global(W: torch.Tensor,
                             lesion_percentage: float,
                             lesion_scale: float,
                             exclude_self_connections: bool,
                             generator: torch.Generator):
        """
        Applies a random lesion to the weight matrix by setting a
        percentage of connections to zero.

        Args:
            lesion_percentage (float): Fraction of existing connections to attenuate (0.0 to 1.0).
            lesion_scale (float): Multiplicative factor applied to selected weights (0.0 to 1.0).
            exclude_self_connections (bool): If True, do not lesion self-connections (i==j).
            generator (torch.Generator): Random number generator.
        """
        if lesion_percentage <= 0.0:
            return W
        if not (0.0 <= lesion_percentage <= 1.0):
            raise ValueError("lesion_percentage must be in [0,1]")
        if not (0.0 <= lesion_scale <= 1.0):
            raise ValueError("lesion_scale must be in [0,1]")

        with torch.no_grad():
            mask_existing = (W != 0)
            if exclude_self_connections:
                diag = torch.eye(W.shape[0], dtype=torch.bool, device=W.device)
                mask_existing = mask_existing & (~diag)
            idx = torch.nonzero(mask_existing, as_tuple=False)
            if idx.numel() == 0:
                return W
            num_to = int(lesion_percentage * idx.shape[0])
            if num_to == 0:
                return W
            perm = torch.randperm(idx.shape[0], generator=generator, device=W.device)[:num_to]
            chosen = idx[perm]
            rows, cols = chosen[:, 0], chosen[:, 1]
            W[rows, cols] = W[rows, cols] * lesion_scale
        return W

    def _apply_lesion_by_type(W: torch.Tensor,
                              exc_np: np.ndarray,
                              inh_np: np.ndarray,
                              lesion_percentage: float,
                              lesion_scale: float,
                              exclude_self_connections: bool,
                              generator: torch.Generator):
        """
        Applies a random lesion to the weight matrix by setting a percentage of connections to zero.

        Args:
            lesion_percentage (float): Fraction of existing connections to attenuate (0.0 to 1.0).
            lesion_scale (float): Multiplicative factor applied to selected weights (0.0 to 1.0).
            exclude_self_connections (bool): If True, do not lesion self-connections (i==j).
            generator (torch.Generator): Random number generator.
        """
        if lesion_percentage <= 0.0:
            return W
        if not (0.0 <= lesion_percentage <= 1.0):
            raise ValueError("lesion_percentage must be in [0,1]")
        if not (0.0 <= lesion_scale <= 1.0):
            raise ValueError("lesion_scale must be in [0,1]")

        with torch.no_grad():
            exc_t = torch.from_numpy(exc_np.astype(bool)).to(W.device)
            inh_t = torch.from_numpy(inh_np.astype(bool)).to(W.device)

            # rows = post, cols = pre
            masks = {
                "E_E": torch.outer(exc_t, exc_t),
                "E_I": torch.outer(inh_t, exc_t),  # to I? careful: rows target; here E_I label means (to I, from E)? 
                "I_E": torch.outer(exc_t, inh_t),
                "I_I": torch.outer(inh_t, inh_t),
            }

            diag = torch.eye(W.shape[0], dtype=torch.bool, device=W.device) if exclude_self_connections else None

            for _, conn_mask in masks.items():
                mask_existing = (W != 0) & conn_mask
                if exclude_self_connections:
                    mask_existing = mask_existing & (~diag)
                idx = torch.nonzero(mask_existing, as_tuple=False)
                if idx.numel() == 0:
                    continue
                num_to = int(lesion_percentage * idx.shape[0])
                if num_to == 0:
                    continue
                perm = torch.randperm(idx.shape[0], generator=generator, device=W.device)[:num_to]
                chosen = idx[perm]
                rows, cols = chosen[:, 0], chosen[:, 1]
                W[rows, cols] = W[rows, cols] * lesion_scale
        return W

    # Apply lesion to W before simulation
    if lesion_percentage > 0.0:
        gen = _rng(device, seed)
        if lesion_by_type:
            W = _apply_lesion_by_type(W, exc, inh, lesion_percentage, lesion_scale, exclude_self_connections, gen)
        else:
            W = _apply_lesion_global(W, lesion_percentage, lesion_scale, exclude_self_connections, gen)

    # Inhibitory and excitatory neurons
    inh_ind = torch.where(torch.tensor(inh, device=device) == 1)[0]
    exc_ind = torch.where(torch.tensor(exc, device=device) == 1)[0]

    # Input stimulus
    u = torch.tensor(u[:, ::downsample], dtype=torch.float64, device=device)
    w_in_tensor = torch.tensor(w_in, dtype=torch.float64, device=device)
    ext_stim = torch.matmul(w_in_tensor, u)

    #------------------------------------------------------
    # LIF network parameters
    #------------------------------------------------------
    dt = 0.00005 * downsample    # sampling rate
    T = (u.shape[1] - 1) * dt * 100  # trial duration (in sec)
    nt = int(np.round(T / dt))           # total number of points in a trial
    tref = 0.002               # refractory time constant (in sec)
    tm = 0.010                 # membrane time constant (in sec)
    vreset = -65               # voltage reset (in mV)
    vpeak = -40                # voltage peak (in mV) for linear LIF

    # Synaptic decay time constants (in sec) for the double-exponential
    # synaptic filter
    # tr: rising time constant
    # td: decay time constants
    # td0: initial decay time constants (before optimization)
    taus_flat = taus.flatten()
    if len(taus_flat) > 1:
        td = (1.0 / (1 + np.exp(-taus_gaus.flatten())) * (taus_flat[1] - taus_flat[0]) + taus_flat[0]) * 5 / 1000
        td0 = (1.0 / (1 + np.exp(-taus_gaus0.flatten())) * (taus_flat[1] - taus_flat[0]) + taus_flat[0]) * 5 / 1000
        tr = 0.002
    else:
        td = taus_flat[0] * 5 / 1000
        td0 = td
        tr = 0.002

    # Convert to scalars if needed
    if np.isscalar(td):
        td = torch.tensor(td, dtype=torch.float64, device=device)
    else:
        td = torch.tensor(td, dtype=torch.float64, device=device)
    
    tr = torch.tensor(tr, dtype=torch.float64, device=device)

    # Synaptic parameters
    IPSC = torch.zeros(N, device=device, dtype=torch.float64)      # post synaptic current storage variable
    h = torch.zeros(N, device=device, dtype=torch.float64)         # storage variable for filtered firing rates
    r = torch.zeros(N, device=device, dtype=torch.float64)         # second storage variable for filtered rates
    hr = torch.zeros(N, device=device, dtype=torch.float64)        # third variable for filtered rates
    JD = torch.zeros(N, device=device, dtype=torch.float64)        # storage variable required for each spike time
    ns = 0                 # number of spikes, counts during simulation

    v = vreset + torch.rand(N, device=device, dtype=torch.float64) * (30 - vreset)  # initialize voltage with random distributions
    v_ = v.clone()   # v_ is the voltage at previous time steps
    v0 = v.clone()   # store the initial voltage values

    # Record REC (membrane voltage), Is (input currents), 
    # spk (spike raster), rs (firing rates) from all the units
    REC = torch.zeros(nt, N, device=device, dtype=torch.float64)  # membrane voltage (in mV) values
    Is = torch.zeros(N, nt, device=device, dtype=torch.float64)  # input currents from the ext_stim
    IPSCs = torch.zeros(N, nt, device=device, dtype=torch.float64) # IPSC over time
    spk = torch.zeros(N, nt, device=device, dtype=torch.float64) # spikes
    rs = torch.zeros(N, nt, device=device, dtype=torch.float64)  # firing rates
    hs = torch.zeros(N, nt, device=device, dtype=torch.float64) # filtered firing rates

    # used to set the refractory times
    tlast = torch.zeros(N, device=device, dtype=torch.float64)

    # Constant bias current to be added to ensure the baseline membrane voltage
    # is around the rheobase
    BIAS = vpeak  # for linear LIF

    #------------------------------------------------------
    # Start the simulation
    #------------------------------------------------------
    for i in range(nt):
        IPSCs[:, i] = IPSC  # record the IPSC over time

        I = IPSC + BIAS  # synaptic current

        # Apply external input stim if there is any
        stim_idx = min(int(i / 100 + 0.5), ext_stim.shape[1] - 1)
        I = I + ext_stim[:, stim_idx]
        Is[:, i] = ext_stim[:, stim_idx]

        # LIF voltage equation with refractory period
        dv = ((dt * (i + 1) > tlast + tref).to(torch.float64) * (-v + I) / tm)  # linear LIF
        v = v + dt * dv + torch.randn(N, device=device, dtype=torch.float64) / 10

        # Artificial stimulation/inhibition
        if 'dur' in stims and stims['mode'].lower() == 'exc':
            if i >= stims['dur'][0] and i < stims['dur'][1]:
                if torch.rand(1).item() < 0.50:
                    v[stims['units']] = v[stims['units']] + 0.5
        elif 'dur' in stims and stims['mode'].lower() == 'inh':
            if i >= stims['dur'][0] and i < stims['dur'][1]:
                if torch.rand(1).item() < 0.50:
                    v[stims['units']] = v[stims['units']] - 0.5

        # find the neurons that have fired
        index = torch.where(v >= vpeak)[0]

        # store spike times, and get the weight matrix column sum of spikers
        if len(index) > 0:
            JD = torch.sum(W[:, index], dim=1)  # compute the increase in current due to spiking
            ns = ns + len(index)  # total number of spikes so far
        else:
            JD = torch.zeros(N, device=device, dtype=torch.float64)

        # used to set the refractory period of LIF neurons
        tlast = tlast + (dt * (i + 1) - tlast) * (v >= vpeak).to(torch.float64)

        # if the rise time is 0, then use the single synaptic filter,
        # otherwise (i.e. rise time is positive) use the double filter
        if tr == 0:
            IPSC = IPSC * torch.exp(-dt / td) + JD * (len(index) > 0) / td
            r = r * torch.exp(-dt / td) + (v >= vpeak).to(torch.float64) / td
            rs[:, i] = r
        else:
            IPSC = IPSC * torch.exp(-dt / td) + h * dt
            h = h * torch.exp(-dt / tr) + JD * (len(index) > 0) / (tr * td)
            hs[:, i] = h

            r = r * torch.exp(-dt / td) + hr * dt
            hr = hr * torch.exp(-dt / tr) + (v >= vpeak).to(torch.float64) / (tr * td)
            rs[:, i] = r

        # record the spikes
        spk[:, i] = (v >= vpeak).to(torch.float64)
        
        v = v + (30 - v) * (v >= vpeak).to(torch.float64)

        # record the membrane voltage tracings from all the units
        REC[i, :] = v

        # reset with spike time interpolant implemented.
        v = v + (vreset - v) * (v >= vpeak).to(torch.float64)

    # Plot the population response
    w_out_tensor = torch.tensor(w_out / scaling_factor, dtype=torch.float64, device=device)
    out = torch.matmul(w_out_tensor, rs)

    # Compute average firing rate for each population (excitatory/inhibitory)
    inh_fr = torch.zeros(len(inh_ind), device=device, dtype=torch.float64)
    for i in range(len(inh_ind)):
        inh_fr[i] = torch.sum(spk[inh_ind[i], :] > 0).to(torch.float64) / T

    exc_fr = torch.zeros(len(exc_ind), device=device, dtype=torch.float64)
    for i in range(len(exc_ind)):
        exc_fr[i] = torch.sum(spk[exc_ind[i], :] > 0).to(torch.float64) / T

    all_fr = torch.zeros(N, device=device, dtype=torch.float64)
    for i in range(N):
        all_fr[i] = torch.sum(spk[i, 10:] > 0).to(torch.float64) / T

    REC = REC.T

    # Some params
    params = {
        'dt': dt,
        'T': T,
        'nt': nt,
        'w_out': w_out,
        'td': td.cpu().numpy() if td.numel() > 1 else td.item(),
        'td0': td0,
        'IPSCs': IPSCs.cpu().numpy()
    }

    return (W.cpu().numpy(), REC.cpu().numpy(), spk.cpu().numpy(), 
            rs.cpu().numpy(), all_fr.cpu().numpy(), out.cpu().numpy().flatten(), params) 