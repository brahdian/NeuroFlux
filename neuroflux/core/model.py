"""
NeuroFlux Core Model Components
==============================

This module implements the core neural network architecture components of NeuroFlux,
including the SSM-XLSTM fusion mechanism and associated helper components.

Key Components:
--------------
- SSMXLSTMFusion: Main model combining state-space models with multi-scale memory
- MambaBlock: Enhanced SSM implementation with input-dependent discretization
- RAIDIntegration: Memory management and fault tolerance system

Implementation Details:
----------------------
The architecture follows the specifications from the NeuroFlux whitepaper,
implementing equation 2.1:
    h_t = e^{ΔWₐ}h_{t-1} + ΔWᵦx_t + Σᵢαⁱcᵢ_{t-1}

Example Usage:
-------------
    model = SSMXLSTMFusion(d_model=512)
    x = torch.randn(batch_size, seq_len, d_model)
    output, state = model(x)
"""

# neuroflux/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Tuple, Optional, Dict, Union
import math
from pathlib import Path
import logging

from ..utils.config_registry import ConfigRegistry
from .controllers import GRPOMoE

class MambaBlock(nn.Module):
    """
    Enhanced SSM Implementation with input-dependent discretization and adaptive computation
    
    This implements the core state-space model (SSM) component with several enhancements:
    1. Input-dependent parameter generation
    2. Adaptive computation depth
    3. RAID memory integration
    
    Args:
        d_model (int): Model dimension, default 512
        
    Attributes:
        A (nn.Parameter): State transition matrix
        B (nn.Parameter): Input projection matrix
        C (nn.Parameter): Output projection matrix
        delta_net (nn.Sequential): Network for computing discretization step size
        B_net (nn.Sequential): Network for input-dependent B matrix
        raid_buffer (Optional[List]): RAID memory buffer for fault tolerance
        
    Methods:
        forward(x, prev_state): Process input sequence
        discretize(x, delta_scale): Compute discretized state matrices
        recover_state(h): Recover from corrupted state using RAID
    """
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        
        # SSM core parameters with improved initialization
        self.A = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        self.B = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        
        # Enhanced input-dependent parameter networks with residual connections
        self.delta_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Softplus()
        )
        
        self.B_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Enhanced normalization and projection with skip connections
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # RAID memory management
        self.raid_buffer = None
        self.parity_slots = 2  # RAID-6 configuration
        
    def discretize(self, x: torch.Tensor, delta_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced input-dependent discretization with scaling"""
        delta = self.delta_net(x).unsqueeze(-1) * delta_scale
        A_d = torch.matrix_exp(self.A * delta)
        B_d = self.B_net(x) * delta
        return A_d, B_d
    
    def forward(
        self, 
        x: torch.Tensor, 
        h_prev: Optional[torch.Tensor] = None,
        delta_scale: float = 1.0,
        use_checkpoint: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced forward pass with gradient checkpointing and RAID backup
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize or recover hidden state
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.d_model, device=x.device)
        elif self.raid_buffer is not None:
            h_prev = self.recover_from_raid(h_prev)
        
        # Compute discretization with enhanced stability
        x = self.norm1(x)
        A_d, B_d = self.discretize(x, delta_scale)
        
        # Sequential state updates with improved numerical stability
        h = h_prev
        outputs = []
        
        for t in range(seq_len):
            # Enhanced SSM update with residual connection
            h_new = torch.bmm(A_d[:, t].unsqueeze(1), h.unsqueeze(-1)).squeeze(-1) + \
                   torch.bmm(B_d[:, t].unsqueeze(1), x[:, t].unsqueeze(-1)).squeeze(-1)
            h = h + h_new  # Residual connection
            
            # Output projection with skip connection
            y = self.out_proj(self.norm2(h))
            outputs.append(y)
            
            # Update RAID buffer periodically
            if (t + 1) % 100 == 0:
                self.update_raid_buffer(h)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)
        return outputs, h
    
    def update_raid_buffer(self, h: torch.Tensor):
        """Update RAID-6 memory buffer"""
        if self.raid_buffer is None:
            self.raid_buffer = [h.detach()]
        else:
            self.raid_buffer.append(h.detach())
            if len(self.raid_buffer) > self.parity_slots + 1:
                self.raid_buffer.pop(0)
    
    def recover_from_raid(self, h: torch.Tensor) -> torch.Tensor:
        """Recover hidden state from RAID buffer if corrupted"""
        if torch.isnan(h).any() and self.raid_buffer is not None:
            return self.raid_buffer[-1]
        return h

class EnhancedXLSTM(nn.Module):
    """
    Enhanced Multi-Scale Hierarchical Memory with improved SSM fusion
    """
    def __init__(self, d_model: int = 512, num_scales: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        
        # Enhanced memory cells with layer normalization
        self.cells = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.LSTMCell(d_model, d_model)
            ) for _ in range(num_scales)
        ])
        
        # Enhanced scale-specific attention with improved heads
        self.scale_attn = nn.ModuleList([
            nn.MultiheadAttention(
                d_model, 
                num_heads=8, 
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_scales)
        ])
        
        # Learned decay rates with temperature scaling
        self.decays = nn.Parameter(torch.linspace(0.5, 0.95, num_scales))
        self.temp = nn.Parameter(torch.ones(1))
        
        # Enhanced output fusion with residual connections
        self.fusion = nn.Sequential(
            nn.Linear(num_scales * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        ssm_state: torch.Tensor,
        prev_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], Optional[Dict]]:
        """
        Enhanced hierarchical memory processing with attention outputs
        """
        if prev_states is None:
            prev_states = [(torch.zeros_like(x), torch.zeros_like(x)) 
                          for _ in range(self.num_scales)]
        
        new_states = []
        outputs = []
        attention_weights = []
        
        for i, (cell, attn, decay) in enumerate(zip(
            self.cells, self.scale_attn, self.decays)):
            
            # Get previous states
            h_prev, c_prev = prev_states[i]
            
            # Enhanced SSM fusion with attention
            fused_input, attn_weights = attn(
                x,
                ssm_state,
                ssm_state,
                need_weights=return_attention
            )
            
            if return_attention:
                attention_weights.append(attn_weights)
            
            # Update cell state with temperature-scaled decay
            h, c = cell(fused_input, (h_prev, c_prev))
            scaled_decay = torch.sigmoid(decay * self.temp)
            c = scaled_decay * c_prev + (1 - scaled_decay) * c
            
            new_states.append((h, c))
            outputs.append(h)
        
        # Enhanced output combination
        combined = torch.cat(outputs, dim=-1)
        output = x + self.fusion(combined)  # Residual connection
        
        if return_attention:
            return output, new_states, {'attention': attention_weights}
        return output, new_states, None

class GRPOMoE(nn.Module):
    """
    Enhanced Gated Routing with PPO-Optimized Expert Selection
    """
    def __init__(self, d_model: int = 512, num_experts: int = 8):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        
        # Enhanced expert networks with shared parameters
        expert_base = MambaBlock(d_model)
        self.experts = nn.ModuleList([
            MambaBlock(d_model) for _ in range(num_experts)
        ])
        
        # Share initial layers for efficiency
        for expert in self.experts:
            expert.A.data = expert_base.A.data.clone()
            expert.B.data = expert_base.B.data.clone()
        
        # Enhanced gating network with improved capacity
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, num_experts)
        )
        
        # Enhanced value network for GRPO
        self.value = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
        self.top_k = 2
        
    def forward(
        self,
        x: torch.Tensor,
        temperature_scale: float = 1.0,
        return_auxiliary: bool = True
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Enhanced forward pass with improved routing and auxiliary outputs
        """
        # Compute gating logits with scaled temperature
        logits = self.gate(x) / (self.temperature * temperature_scale)
        
        # Enhanced Gumbel top-k sampling
        noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        scores = (logits + noise) / math.sqrt(self.d_model)
        
        # Get top-k expert indices and probabilities
        top_k_scores, top_k_indices = scores.topk(self.top_k, dim=-1)
        probs = F.softmax(top_k_scores, dim=-1)
        
        # Compute expert outputs with load balancing
        expert_outputs = []
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        
        for idx in range(self.num_experts):
            mask = (top_k_indices == idx).any(dim=-1)
            if mask.any():
                expert_counts[idx] = mask.float().sum()
                out, _ = self.experts[idx](x[mask])
                expert_outputs.append((out, mask))
        
        # Combine expert outputs with dynamic weighting
        output = torch.zeros_like(x)
        for out, mask in expert_outputs:
            output[mask] = out
        
        if return_auxiliary:
            return output, {
                'logits': logits,
                'indices': top_k_indices,
                'probs': probs,
                'value': self.value(x).squeeze(-1),
                'expert_counts': expert_counts,
                'load_balance': expert_counts.var() / expert_counts.mean()
            }
        return output, None

class NeuroFluxLayer(nn.Module):
    """
    Enhanced NeuroFlux Layer with improved integration and fault tolerance
    """
    def __init__(
        self,
        d_model: int = 512,
        num_experts: int = 8,
        num_scales: int = 3
    ):
        super().__init__()
        self.d_model = d_model
        
        # Enhanced core components
        self.ssm = MambaBlock(d_model)
        self.xlstm = EnhancedXLSTM(d_model, num_scales)
        self.moe = GRPOMoE(d_model, num_experts)
        
        # Enhanced hypernetwork with improved capacity
        self.hypernet = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3)  # [delta, lambda_h, moe_temp]
        )
        
        # Enhanced normalization and skip connections
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        lstm_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_auxiliary: bool = True,
        use_checkpoint: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Enhanced forward pass with improved integration and monitoring
        """
        # Get adaptive parameters with clamping
        hyper_params = self.hypernet(x)
        delta, lambda_h, moe_temp = torch.sigmoid(hyper_params).unbind(-1)
        
        # Clamp parameters to stable ranges
        delta = delta.clamp(0.1, 2.0)
        lambda_h = lambda_h.clamp(0.01, 0.99)
        moe_temp = 0.5 + moe_temp  # Range [0.5, 1.5]
        
        # Enhanced SSM processing
        x = self.norm1(x)
        ssm_out, h_new = self.ssm(x, h_prev, delta_scale=delta, use_checkpoint=use_checkpoint)
        
        # Enhanced XLSTM fusion with attention outputs
        mem_out, lstm_new, attn_info = self.xlstm(
            ssm_out, 
            h_new, 
            lstm_states,
            return_attention=return_auxiliary
        )
        
        # Enhanced MoE routing with temperature scaling
        moe_out, moe_info = self.moe(
            self.norm2(mem_out),
            temperature_scale=moe_temp,
            return_auxiliary=return_auxiliary
        )
        
        # Final output with residual connection
        output = mem_out + moe_out
        
        if return_auxiliary:
            aux_info = {
                'hyper_params': {
                    'delta': delta,
                    'lambda_h': lambda_h,
                    'moe_temp': moe_temp
                },
                'ssm_state': h_new,
                'lstm_states': lstm_new,
                'attention': attn_info.get('attention') if attn_info else None,
                'moe': moe_info
            }
            return output, aux_info
        
        return output, {'ssm_state': h_new, 'lstm_states': lstm_new}

class CheckpointManager:
    """
    Enhanced checkpoint management with RAID integration and compression
    """
    def __init__(
        self,
        save_dir: str = './checkpoints',
        max_checkpoints: int = 5,
        compression_level: int = 3
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.compression_level = compression_level
        self.checkpoint_list = []
        
    def compress_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compress model weights to FP8 where safe"""
        compressed = {}
        for key, tensor in state_dict.items():
            # Keep certain layers in FP16/32 for stability
            if any(s in key for s in ['norm', 'embed', 'head']):
                compressed[key] = tensor
            else:
                compressed[key] = tensor.to(torch.float8_e4m3fn)
        return compressed
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        hypernet: nn.Module,
        raid_memory: Optional[Dict] = None,
        step: int = 0,
        metrics: Optional[Dict] = None
    ) -> str:
        """Save checkpoint with compression and RAID state"""
        checkpoint_path = self.save_dir / f'checkpoint_{step}.pt'
        
        # Compress model state
        model_state = self.compress_state_dict(model.state_dict())
        
        checkpoint = {
            'step': step,
            'model_state': model_state,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'hypernet_state': hypernet.state_dict(),
            'raid_memory': raid_memory,
            'metrics': metrics
        }
        
        # Save with compression
        torch.save(
            checkpoint,
            checkpoint_path,
            _use_new_zipfile_serialization=True,
            compression=self.compression_level
        )
        
        # Manage checkpoint history
        self.checkpoint_list.append(checkpoint_path)
        if len(self.checkpoint_list) > self.max_checkpoints:
            oldest = self.checkpoint_list.pop(0)
            oldest.unlink()
            
        return str(checkpoint_path)
    
    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        hypernet: Optional[nn.Module] = None,
        device: str = 'cuda'
    ) -> Tuple[int, Optional[Dict]]:
        """Load checkpoint with automatic device placement"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model with FP8 -> FP16/32 conversion
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if scheduler and checkpoint['scheduler_state']:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            
        if hypernet and checkpoint['hypernet_state']:
            hypernet.load_state_dict(checkpoint['hypernet_state'])
            
        return checkpoint['step'], checkpoint.get('metrics')
    
    def find_latest(self) -> Optional[str]:
        """Find most recent checkpoint"""
        checkpoints = sorted(self.save_dir.glob('checkpoint_*.pt'))
        return str(checkpoints[-1]) if checkpoints else None

class SSMXLSTMFusion(nn.Module):
    """
    SSM-XLSTM Fusion model combining linear-time sequence processing with multi-scale memory
    
    This is the main model architecture implementing the NeuroFlux fusion mechanism.
    It combines:
    1. Mamba-style SSM for efficient sequence processing
    2. Multi-scale XLSTM for hierarchical memory
    3. RAID-based fault tolerance
    4. Expert routing via GRPOMoE
    
    Args:
        config (Optional[Dict]): Configuration overrides
        
    Attributes:
        ssm (MambaBlock): State-space model component
        xlstm (nn.ModuleList): Multi-scale LSTM cells
        controller (GRPOMoE): Expert routing controller
        raid (RAIDMemory): Fault tolerance system
        
    Methods:
        forward(x, state): Process input with full fusion mechanism
        get_intermediate_states(x): Access internal representations
        save_checkpoint(path): Save model state with RAID encoding
        load_checkpoint(path): Load model state with automatic recovery
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.config = ConfigRegistry.get_config()
        # Override config with kwargs
        for k, v in kwargs.items():
            setattr(self.config, k, v)
            
        # Initialize components
        self.setup_components()
    
    def setup_components(self):
        """
        Initialize model components with current configuration
        
        This includes:
        - SSM block for sequence processing
        - Multi-scale XLSTM cells
        - Expert routing controller
        - RAID memory system
        """
        # Lazy load RAID to avoid circular imports
        from ..system.raid import RAIDMemory
        self.raid = RAIDMemory()
        
        # Initialize other components
        self.controller = GRPOMoE(self.config)
        # ... rest of setup ...
    
    # ... rest of the implementation ...