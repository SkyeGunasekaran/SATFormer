from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperConnection(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        expansion_rate: int = 4,
        layer_idx: int = 0,
        dynamic: bool = True,
        use_tanh: bool = True,
        norm_type: str = 'rms',
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.expansion_rate = expansion_rate
        self.layer_idx = layer_idx
        self.dynamic = dynamic
        self.use_tanh = use_tanh
        
        n = expansion_rate
        
        # Static parameters (B, Am, Ar)
        # Initialize B as ones (n,)
        self.static_beta = nn.Parameter(torch.ones(n))
        
        # Initialize Am and Ar according to paper's initialization (Eq. 14)
        # Am is initialized as e_{k mod n} (one-hot based on layer index)
        # Ar is initialized as identity matrix
        init_alpha_m = torch.zeros(n, 1)
        init_alpha_m[layer_idx % n, 0] = 1.0
        
        # Combine Am and Ar into a single parameter for efficiency
        # alpha has shape (n, n+1) where first column is Am and rest is Ar
        init_alpha = torch.cat([init_alpha_m, torch.eye(n)], dim=1)
        self.static_alpha = nn.Parameter(init_alpha)
        
        if dynamic:
            # Dynamic parameters
            # Normalization layer (before computing dynamic weights)
            if norm_type == 'layer':
                self.norm = nn.LayerNorm(hidden_size)
            else:
                self.norm = nn.RMSNorm(hidden_size)
            
            self.dynamic_fused_proj = nn.Parameter(torch.zeros(hidden_size, n + 1 + 1))
            
            # Small initial scales for dynamic components (s_beta, s_alpha in paper)
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
    
    def _compute_weights(
        self,
        hyper_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.dynamic:
            # Normalize the hyper hidden states
            norm_h = self.norm(hyper_hidden)  # (B, L, N, D)
            
            # (B, L, N, D) @ (D, N+2) -> (B, L, N, N+2)
            dynamic_fused = torch.matmul(norm_h, self.dynamic_fused_proj)
            
            # Split: column 0 = beta, columns 1: = alpha
            dynamic_beta = dynamic_fused[..., 0]        # (B, L, N)
            dynamic_alpha = dynamic_fused[..., 1:]       # (B, L, N, N+1)
            
            if self.use_tanh:
                dynamic_alpha = torch.tanh(dynamic_alpha)
                dynamic_beta = torch.tanh(dynamic_beta)
            
            # alpha = static_alpha + dynamic_alpha * scale
            alpha = torch.addcmul(
                self.static_alpha.unsqueeze(0).unsqueeze(0),
                dynamic_alpha,
                self.dynamic_alpha_scale,
            )
            # beta = static_beta + dynamic_beta * scale
            beta = torch.addcmul(
                self.static_beta.unsqueeze(0).unsqueeze(0),
                dynamic_beta,
                self.dynamic_beta_scale,
            )
        else:
            # Static weights only
            alpha = self.static_alpha.unsqueeze(0).unsqueeze(0)
            beta = self.static_beta.unsqueeze(0).unsqueeze(0)
        
        return alpha, beta
    
    def width_connection(
        self,
        hyper_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        alpha, beta = self._compute_weights(hyper_hidden)
        
        mix_h = torch.einsum('blnm,blnd->blmd', alpha, hyper_hidden)
        
        return mix_h, beta
    
    def depth_connection(
        self,
        mix_h: torch.Tensor,
        layer_output: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:

        # H' = mix_h[:, :, 1:, :], shape (B, L, N, D)
        h_prime = mix_h[:, :, 1:, :]
        
        hyper_hidden = torch.einsum('bln,bld->blnd', beta, layer_output)
        hyper_hidden += h_prime
        
        return hyper_hidden
    
    def forward(
        self,
        hyper_hidden: torch.Tensor,
        layer_fn: callable,
        **layer_kwargs,
    ) -> Tuple[torch.Tensor, any]:

        # Width connection
        mix_h, beta = self.width_connection(hyper_hidden)
        
        # Extract layer input (h_0)
        layer_input = mix_h[:, :, 0, :]  # (B, L, D)
        
        # Apply layer
        layer_outputs = layer_fn(layer_input, **layer_kwargs)
        
        if isinstance(layer_outputs, tuple):
            layer_output = layer_outputs[0]
            extra_outputs = layer_outputs[1:]
        else:
            layer_output = layer_outputs
            extra_outputs = None
        
        # Depth connection
        hyper_hidden = self.depth_connection(mix_h, layer_output, beta)
        
        if extra_outputs is not None:
            return hyper_hidden, extra_outputs
        return hyper_hidden, None


def expand_to_hyper_hidden(
    hidden_states: torch.Tensor,
    expansion_rate: int,
) -> torch.Tensor:
    
    return hidden_states.unsqueeze(-2).repeat(1, 1, expansion_rate, 1)


def collapse_hyper_hidden(
    hyper_hidden: torch.Tensor,
) -> torch.Tensor:
    return hyper_hidden.sum(dim=-2)