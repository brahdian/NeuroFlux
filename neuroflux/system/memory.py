# neuroflux/memory.py
import torch
import torch.nn as nn
import math
from typing import Optional, List, Tuple

class EnhancedXLSTM(nn.Module):
    """
    Enhanced Multi-Scale Hierarchical Memory Cell with complete implementation
    of the whitepaper's XLSTM fusion mechanism
    """
    def __init__(self, d_model=512, scales=3):
        super().__init__()
        self.d_model = d_model
        self.scales = scales
        
        # Multi-scale memory cells
        self.cells = nn.ModuleList([
            nn.LSTMCell(d_model, d_model) for _ in range(scales)
        ])
        
        # Learnable decay rates per scale
        self.decays = nn.Parameter(torch.linspace(0.5, 0.9, scales))
        
        # Scale-specific transformations
        self.scale_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            ) for _ in range(scales)
        ])
        
        # Memory fusion
        self.fusion = nn.MultiheadAttention(d_model, 8, dropout=0.1)
        self.mix = nn.Linear(scales * d_model, d_model)
        
    def forward(self, x, prev_states, ssm_state):
        """
        Implements the complete SSM-XLSTM fusion from the whitepaper
        x: (B, D)
        prev_states: List[(h, c)]
        ssm_state: (B, D) from SSM
        """
        new_states = []
        outputs = []
        
        for i, (cell, decay, transform) in enumerate(zip(
            self.cells, self.decays, self.scale_transforms)):
            
            # Get previous states or initialize
            h_prev, c_prev = prev_states[i] if prev_states else (x, torch.zeros_like(x))
            
            # Scale-specific processing
            x_scaled = transform(x)
            
            # LSTM with temporal decay
            h, c = cell(x_scaled, (h_prev, c_prev))
            c = decay * c_prev + (1 - decay) * c
            
            # Store states
            new_states.append((h, c))
            outputs.append(h)
        
        # Concatenate all scales
        multi_scale = torch.cat(outputs, dim=-1)
        
        # Fuse with SSM state via attention
        fused_mem, _ = self.fusion(
            ssm_state.unsqueeze(0),
            multi_scale.unsqueeze(0),
            multi_scale.unsqueeze(0)
        )
        
        return self.mix(multi_scale) + fused_mem.squeeze(0), new_states

class SSMXLSTMFusion(nn.Module):
    """
    Implementation of SSM-XLSTM Fusion from NeuroFlux whitepaper Section 2.1
    Combines linear-time SSM processing with multi-scale memory
    """
    def __init__(self, d_model: int = 512, num_scales: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        
        # SSM core parameters
        self.A = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        self.B = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        
        # Input-dependent discretization network
        self.delta_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Softplus()  # Ensures positive Δ
        )
        
        # Multi-scale XLSTM cells
        self.xlstm_cells = nn.ModuleList([
            nn.LSTMCell(d_model, d_model)
            for _ in range(num_scales)
        ])
        
        # Scale-specific decay rates (learnable)
        self.decays = nn.Parameter(torch.linspace(0.5, 0.95, num_scales))
        
        # Memory fusion attention
        self.fusion_attn = nn.MultiheadAttention(
            d_model, 
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model * (num_scales + 1), d_model),
            nn.LayerNorm(d_model)
        )
        
        # RAID memory management
        self.raid_buffer = None
        
    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        c_prev: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass implementing equation from whitepaper:
        h_t = e^{ΔWₐ}h_{t-1} + ΔWᵦx_t + Σᵢαⁱcᵢ_{t-1}
        
        Args:
            x: Input tensor (batch_size, d_model)
            h_prev: Previous SSM state
            c_prev: Previous XLSTM states [(h, c)] for each scale
        """
        batch_size = x.size(0)
        
        # Initialize states if needed
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.d_model, device=x.device)
        if c_prev is None:
            c_prev = [(torch.zeros_like(h_prev), torch.zeros_like(h_prev)) 
                     for _ in range(self.num_scales)]
            
        # Compute input-dependent discretization (Δ)
        delta = self.delta_net(x)
        
        # SSM state update
        A_discrete = torch.matrix_exp(delta * self.A)
        h_ssm = torch.bmm(A_discrete.unsqueeze(1), h_prev.unsqueeze(-1)).squeeze(-1) + \
                torch.bmm(delta * self.B, x.unsqueeze(-1)).squeeze(-1)
                
        # XLSTM updates with temporal decay
        c_new = []
        h_scales = []
        
        for i, (lstm_cell, decay) in enumerate(zip(self.xlstm_cells, self.decays)):
            h_i, c_i = c_prev[i]
            
            # LSTM update
            h_next, c_next = lstm_cell(x, (h_i, c_i))
            
            # Apply temporal decay to cell state
            c_next = decay * c_i + (1 - decay) * c_next
            
            c_new.append((h_next, c_next))
            h_scales.append(h_next)
            
        # Memory fusion via attention
        h_scales_cat = torch.stack(h_scales, dim=1)  # [batch, num_scales, d_model]
        h_ssm_expanded = h_ssm.unsqueeze(1)  # [batch, 1, d_model]
        
        fused_mem, _ = self.fusion_attn(
            h_ssm_expanded,
            h_scales_cat,
            h_scales_cat
        )
        
        # Combine all representations
        combined = torch.cat([
            h_ssm,
            fused_mem.squeeze(1),
            *h_scales
        ], dim=-1)
        
        output = self.out_proj(combined)
        
        # Update RAID buffer periodically
        if self.training and torch.rand(1).item() < 0.01:  # 1% chance
            self._update_raid_buffer(h_ssm)
            
        return output, h_ssm, c_new
        
    def _update_raid_buffer(self, h: torch.Tensor):
        """Update RAID memory buffer for fault tolerance"""
        if self.raid_buffer is None:
            self.raid_buffer = [h.detach()]
        else:
            self.raid_buffer.append(h.detach())
            if len(self.raid_buffer) > 3:  # Keep last 3 states
                self.raid_buffer.pop(0)
                
    def recover_state(self, h: torch.Tensor) -> torch.Tensor:
        """Recover from corrupted state using RAID buffer"""
        if torch.isnan(h).any() and self.raid_buffer:
            return self.raid_buffer[-1]
        return h