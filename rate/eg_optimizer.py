"""Exponentiated Gradient Optimizer"""

import torch

class ExponentiatedGradient(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, weight_decay=0.0,
                 eps=1e-12, max_norm=None, norm_axis=None):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                        eps=eps, max_norm=max_norm, norm_axis=norm_axis)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, wd, eps = group['lr'], group['weight_decay'], group['eps']
            max_norm, norm_axis = group['max_norm'], group['norm_axis']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad + wd * p if wd else p.grad
                update = torch.exp(-lr * g).clamp_min(torch.finfo(p.dtype).tiny)
                p.mul_(update).clamp_min_(eps)
                # optional per‑column norm control
                if max_norm is not None:
                    n = p.norm(dim=norm_axis, keepdim=True)
                    scale = (max_norm / (n + 1e-12)).clamp_max(1.0)
                    p.mul_(scale)