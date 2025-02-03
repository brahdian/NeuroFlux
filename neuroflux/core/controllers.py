import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sortedcontainers import SortedList
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from einops import rearrange

from ..utils.config_registry import ConfigRegistry
from .hypernetwork import DifferentiableHyperNetwork

@dataclass
class Expert(nn.Module):
    """
    Expert module implementing the enhanced architecture from Section 2.3
    Features:
    - Adaptive computation depth
    - Residual memory integration
    - Dynamic feature transformation
    """
    def __init__(self, d_model: int, d_ff: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Multi-scale feature transformation
        self.input_proj = nn.Linear(d_model, d_ff)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ff1': nn.Linear(d_ff, d_ff * 2),
                'ff2': nn.Linear(d_ff * 2, d_ff),
                'norm': nn.LayerNorm(d_ff),
                'gate': nn.Linear(d_ff, 1, bias=False)
            }) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_ff, d_model)
        
        # Memory integration
        self.memory_key = nn.Linear(d_ff, d_model)
        self.memory_value = nn.Linear(d_ff, d_model)
        self.memory_query = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with adaptive computation and memory integration
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            memory: Optional memory tensor [batch_size, mem_len, d_model]
        Returns:
            output: Processed tensor [batch_size, seq_len, d_model]
            memory: Updated memory tensor [batch_size, mem_len, d_model]
        """
        # Initial projection
        h = self.input_proj(x)
        
        # Adaptive layer processing
        layer_outputs = []
        for layer in self.layers:
            # Feature transformation
            h2 = layer['ff1'](h)
            h2 = F.gelu(h2)
            h2 = layer['ff2'](h2)
            h2 = layer['norm'](h2)
            
            # Adaptive gating
            gate = torch.sigmoid(layer['gate'](h))
            h = h + self.dropout(gate * h2)
            layer_outputs.append(h)
        
        # Memory integration
        if memory is not None:
            query = self.memory_query(x)
            key = self.memory_key(h)
            value = self.memory_value(h)
            
            # Compute attention
            scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model ** 0.5)
            attn = F.softmax(scores, dim=-1)
            memory_output = torch.matmul(attn, value)
            
            # Update memory
            new_memory = torch.cat([memory[:, 1:], h[:, -1:]], dim=1)
            
            # Combine with transformed features
            h = h + self.dropout(memory_output)
        else:
            new_memory = h[:, -1:]
        
        # Final projection and residual connection
        output = self.output_proj(h)
        output = self.norm(x + self.dropout(output))
        
        return output, new_memory

class EnhancedGRPOMoEController(nn.Module):
    """
    Enhanced GRPO Controller with dynamic KL-constrained updates and adaptive clipping
    Implements Sections 2.2(GRPO-MoE) and 3.2(RL-Driven Controllers) from whitepaper
    """
    def __init__(
        self, 
        input_dim: int = 4, 
        max_k: int = 8,
        initial_temp: float = 1.0,
        min_entropy: float = -4.0
    ):
        super().__init__()
        self.max_k = max_k
        
        # Dynamic clip ratio with gradient stopping
        self.clip_ratio = nn.Parameter(torch.tensor(0.2))
        self.clip_ratio.register_hook(lambda grad: grad * 0.0)
        
        # Enhanced trust region with uncertainty estimation
        self.trust_region = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 2)  # [mean, log_std]
        )
        
        # Enhanced actor network with residual connections
        self.actor_base = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.actor_residual = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256)
        )
        self.actor_head = nn.Linear(256, max_k)
        
        # Dual-headed critic for uncertainty-aware value estimation
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 3)  # [value, uncertainty, advantage]
        )
        
        # Adaptive temperature control
        self.log_temp = nn.Parameter(torch.log(torch.tensor(initial_temp)))
        self.target_entropy = nn.Parameter(torch.tensor(min_entropy))
        
        # KL tracking buffers with improved statistics
        self.register_buffer('kl_mvg_avg', torch.zeros(1))
        self.register_buffer('kl_mvg_std', torch.ones(1))
        self.register_buffer('kl_min', torch.ones(1) * float('inf'))
        self.register_buffer('kl_max', torch.zeros(1))
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enhanced forward pass with improved gradient flow"""
        # Actor forward with residual connection
        base_features = self.actor_base(state)
        residual = self.actor_residual(base_features)
        features = base_features + residual
        policy_logits = self.actor_head(features)
        
        # Critic forward with uncertainty
        value_out = self.critic(state)
        value, uncertainty, advantage = value_out.unbind(-1)
        
        return policy_logits, value, uncertainty, advantage

    def get_action(
        self, 
        state: torch.Tensor,
        training: bool = True,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor, Dict]:
        """
        Enhanced action selection with improved exploration
        Returns: (action, log_prob, value, auxiliary_info)
        """
        logits, value, uncertainty, advantage = self.forward(state)
        
        if deterministic or not training:
            action = logits.argmax(dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)[action]
            return action.item(), log_prob, value, {
                'uncertainty': uncertainty,
                'advantage': advantage
            }
        
        # Trust region parameters
        tr_mean, tr_log_std = self.trust_region(state).unbind(-1)
        tr_std = torch.exp(tr_log_std.clamp(-5, 2))
        
        # Temperature-scaled exploration
        temp = torch.exp(self.log_temp).clamp(0.1, 5.0)
        
        # Gumbel-Top-k sampling with trust region
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        scaled_noise = gumbel_noise * tr_std.unsqueeze(-1)
        noisy_logits = logits / temp + scaled_noise
        
        # Dynamic k selection
        k = max(2, int((tr_mean.sigmoid() * self.max_k).item()))
        top_k_logits, indices = noisy_logits.topk(k)
        
        # Normalized probabilities for selected actions
        probs = F.softmax(top_k_logits / temp, dim=-1)
        action_idx = torch.multinomial(probs, 1).squeeze()
        action = indices[action_idx]
        
        log_prob = F.log_softmax(logits, dim=-1)[action]
        
        return action.item(), log_prob, value, {
            'uncertainty': uncertainty,
            'advantage': advantage,
            'temperature': temp.item(),
            'trust_region': {
                'mean': tr_mean.item(),
                'std': tr_std.item()
            }
        }

    def update_kl_stats(
        self, 
        old_log_probs: torch.Tensor, 
        new_log_probs: torch.Tensor
    ):
        """Enhanced KL tracking with improved statistics"""
        kl_div = (old_log_probs.exp() * (old_log_probs - new_log_probs)).sum(-1)
        
        # Update running statistics
        new_avg = 0.95 * self.kl_mvg_avg + 0.05 * kl_div.mean()
        new_std = 0.95 * self.kl_mvg_std + 0.05 * kl_div.std()
        
        # Update min/max
        self.kl_min.copy_(torch.min(self.kl_min, kl_div.min()))
        self.kl_max.copy_(torch.max(self.kl_max, kl_div.max()))
        
        # Update moving averages
        self.kl_mvg_avg.copy_(new_avg)
        self.kl_mvg_std.copy_(new_std)
        
        return {
            'kl_div': kl_div.mean().item(),
            'kl_std': kl_div.std().item(),
            'kl_min': self.kl_min.item(),
            'kl_max': self.kl_max.item()
        }

    def compute_loss(
        self,
        old_log_probs: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced loss computation with improved PPO objectives
        """
        # Get current policy and value estimates
        logits, curr_value, uncertainty, _ = self.forward(state)
        new_log_probs = F.log_softmax(logits, dim=-1).gather(1, actions)
        
        # Dynamic clip ratio based on KL statistics
        clip_ratio = self.dynamic_clip_ratio()
        
        # Policy loss with adaptive clipping
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
        
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # Value loss with uncertainty weighting
        value_pred_clipped = values + (curr_value - values).clamp(-clip_ratio, clip_ratio)
        value_losses = (curr_value - returns) ** 2
        value_losses_clipped = (value_pred_clipped - returns) ** 2
        
        uncertainty_weight = torch.exp(-uncertainty).detach()
        value_loss = (torch.min(value_losses, value_losses_clipped) * uncertainty_weight).mean()
        
        # Temperature loss for adaptive exploration
        temp = torch.exp(self.log_temp)
        curr_entropy = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        temp_loss = temp * (curr_entropy - self.target_entropy).detach()
        
        # Trust region loss
        tr_mean, tr_log_std = self.trust_region(state).unbind(-1)
        tr_loss = (tr_log_std.exp() + tr_mean.abs()).mean() * 0.1
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'temperature_loss': temp_loss,
            'trust_region_loss': tr_loss,
            'total_loss': policy_loss + 0.5 * value_loss + temp_loss + tr_loss,
            'metrics': {
                'entropy': curr_entropy.item(),
                'clip_ratio': clip_ratio,
                'temperature': temp.item(),
                'value_uncertainty': uncertainty.mean().item()
            }
        }

class EnhancedRAIDController(nn.Module):
    """
    Enhanced RAID controller with predictive parity allocation
    Implements Section 4.1's adaptive parity allocation with non-uniform coding
    """
    def __init__(
        self, 
        input_dim: int = 5, 
        max_parity: int = 6,
        min_parity: int = 2
    ):
        super().__init__()
        self.max_parity = max_parity
        self.min_parity = min_parity
        
        # Enhanced parity allocation network
        self.parity_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, max_parity + 1)  # +1 for mean parity prediction
        )
        
        # Enhanced failure prediction with uncertainty
        self.failure_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 2)  # [prob, uncertainty]
        )
        
        # Value network with extended state
        self.value_net = nn.Sequential(
            nn.Linear(input_dim + max_parity + 3, 64),  # +3 for failure prob, uncertainty, mean parity
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with improved parity allocation"""
        # Get parity distribution and mean
        parity_out = self.parity_net(state)
        parity_logits = parity_out[:, :-1]
        mean_parity = parity_out[:, -1].sigmoid() * (self.max_parity - self.min_parity) + self.min_parity
        
        # Get failure prediction with uncertainty
        failure_pred = self.failure_predictor(state)
        failure_prob, failure_uncertainty = failure_pred.unbind(-1)
        failure_prob = failure_prob.sigmoid()
        failure_uncertainty = failure_uncertainty.exp().clamp(0.1, 10.0)
        
        # Compute parity weights with failure-aware scaling
        parity_weights = F.softmax(parity_logits, dim=-1)
        scaled_weights = parity_weights * (1 + failure_prob.unsqueeze(-1))
        
        # Generate Poisson samples for non-uniform allocation
        allocation = torch.poisson(scaled_weights * mean_parity.unsqueeze(-1))
        parity_allocation = torch.clamp(allocation, self.min_parity, self.max_parity).int()
        
        # Compute value estimate
        value_input = torch.cat([
            state,
            parity_weights,
            failure_prob.unsqueeze(-1),
            failure_uncertainty.unsqueeze(-1),
            mean_parity.unsqueeze(-1)
        ], dim=-1)
        value = self.value_net(value_input)
        
        return {
            'parity_allocation': parity_allocation,
            'parity_weights': parity_weights,
            'mean_parity': mean_parity,
            'failure_prob': failure_prob,
            'failure_uncertainty': failure_uncertainty,
            'value': value
        }

class AdaptiveControllerManager:
    """
    Enhanced experience management with curriculum-aware prioritization
    Implements Section 3.2's Differentiable Auto-Tuning
    """
    def __init__(
        self,
        capacity: int = 100000,
        total_steps: int = 1000000,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6
    ):
        self.sum_tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.total_steps = total_steps
        
        # Enhanced phase transitions with overlap
        self.phase_boundaries = {
            'exploration': (0.0, 0.35),
            'exploitation': (0.3, 0.65),
            'consolidation': (0.6, 1.0)
        }
        
        # Statistics tracking
        self.max_priority = 1.0
        self.min_priority = 1.0
        self.priority_stats = {
            'mean': 0.0,
            'std': 1.0,
            'count': 0
        }
        
        # Experience tracking
        self.total_experiences = 0
        self.current_step = 0
        
    def store_experience(
        self, 
        experience: Dict,
        priority: Optional[float] = None
    ):
        """Enhanced experience storage with statistics tracking"""
        if priority is None:
            priority = self.max_priority
            
        # Update priority statistics
        self.max_priority = max(self.max_priority, priority)
        self.min_priority = min(self.min_priority, priority)
        
        # Update running statistics
        self.priority_stats['count'] += 1
        count = self.priority_stats['count']
        delta = priority - self.priority_stats['mean']
        self.priority_stats['mean'] += delta / count
        delta2 = priority - self.priority_stats['mean']
        self.priority_stats['std'] = np.sqrt(
            (self.priority_stats['std']**2 * (count - 1) + delta * delta2) / count
        )
        
        # Store with phase-aware priority scaling
        phase_factor = self._get_phase_factor()
        scaled_priority = (priority ** self.alpha) * phase_factor
        self.sum_tree.add(max(scaled_priority, self.eps), experience)
        self.total_experiences += 1
        
    def sample_batch(
        self, 
        batch_size: int,
        beta: Optional[float] = None
    ) -> Tuple[List, List, np.ndarray]:
        """Enhanced batch sampling with adaptive importance sampling"""
        if beta is None:
            beta = self.beta
            
        # Compute adaptive segment size
        segment = self.sum_tree.total() / batch_size
        
        # Initialize batch data
        samples = []
        indices = []
        priorities = []
        
        # Sample with stratification
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            # Add noise for exploration
            mass = np.random.uniform(a, b)
            
            # Get sample from sum tree
            idx, priority, experience = self.sum_tree.get(mass)
            
            samples.append(experience)
            indices.append(idx)
            priorities.append(priority)
            
        # Compute importance sampling weights
        priorities = np.array(priorities)
        probs = priorities / self.sum_tree.total()
        
        # Phase-aware importance sampling
        phase_factor = self._get_phase_factor()
        weights = (len(self.sum_tree) * probs * phase_factor) ** -beta
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(
        self, 
        indices: List[int],
        priorities: List[float],
        error_scale: float = 1.0
    ):
        """Update priorities with phase-dependent scaling"""
        phase = self._current_phase()
        
        for idx, priority in zip(indices, priorities):
            # Scale priority based on current phase
            if phase == 'exploration':
                # Encourage exploration of high-error experiences
                scaled_priority = priority * 1.2
            elif phase == 'exploitation':
                # Focus on experiences with moderate errors
                scaled_priority = priority
            else:  # consolidation
                # Prioritize low-error experiences
                scaled_priority = priority * 0.8
                
            # Apply error scaling and clipping
            final_priority = np.clip(
                scaled_priority * error_scale,
                self.min_priority,
                self.max_priority
            )
            
            # Update sum tree
            self.sum_tree.update(idx, (final_priority + self.eps) ** self.alpha)
    
    def _current_phase(self) -> str:
        """Determine current curriculum phase with smooth transitions"""
        progress = self.current_step / self.total_steps
        
        # Check each phase boundary
        for phase, (start, end) in self.phase_boundaries.items():
            if start <= progress <= end:
                return phase
                
        return 'consolidation'  # Default to consolidation
    
    def _get_phase_factor(self) -> float:
        """Calculate phase-dependent priority scaling factor"""
        progress = self.current_step / self.total_steps
        phase = self._current_phase()
        
        if phase == 'exploration':
            # Higher factor for exploration
            return 1.2 - 0.4 * (progress / 0.35)
        elif phase == 'exploitation':
            # Balanced factor for exploitation
            return 1.0
        else:
            # Lower factor for consolidation
            return 0.8 + 0.2 * ((progress - 0.6) / 0.4)
    
    def step(self):
        """Update internal step counter"""
        self.current_step += 1
        
    def get_stats(self) -> Dict:
        """Get current statistics"""
        return {
            'total_experiences': self.total_experiences,
            'current_phase': self._current_phase(),
            'phase_factor': self._get_phase_factor(),
            'priority_stats': {
                'mean': self.priority_stats['mean'],
                'std': self.priority_stats['std'],
                'min': self.min_priority,
                'max': self.max_priority
            },
            'progress': self.current_step / self.total_steps
        }
    
    def save_state(self) -> Dict:
        """Save manager state for checkpointing"""
        return {
            'priority_stats': self.priority_stats,
            'max_priority': self.max_priority,
            'min_priority': self.min_priority,
            'total_experiences': self.total_experiences,
            'current_step': self.current_step,
            'tree_data': self.sum_tree.get_state()
        }
    
    def load_state(self, state: Dict):
        """Load manager state from checkpoint"""
        self.priority_stats = state['priority_stats']
        self.max_priority = state['max_priority']
        self.min_priority = state['min_priority']
        self.total_experiences = state['total_experiences']
        self.current_step = state['current_step']
        self.sum_tree.set_state(state['tree_data'])

class SumTree:
    """Enhanced SumTree with improved state management and statistics"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
        
        # Add tracking for min/max values
        self.max_recorded = float('-inf')
        self.min_recorded = float('inf')
    
    def _propagate(self, idx: int, change: float):
        """Propagate value change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
            
    def _retrieve(self, idx: int, s: float) -> int:
        """Find the leaf index containing the given cumulative sum"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
            
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
            
    def add(self, p: float, data: Any):
        """Add new data with priority p"""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        if self.n_entries < self.capacity:
            self.n_entries += 1
            
        # Update statistics
        self.max_recorded = max(self.max_recorded, p)
        self.min_recorded = min(self.min_recorded, p)
        
    def update(self, idx: int, p: float):
        """Update node priority and propagate changes"""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
        
    def get(self, s: float) -> Tuple[int, float, Any]:
        """Get item based on cumulative sum s"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        return (idx, self.tree[idx], self.data[dataIdx])
        
    def total(self) -> float:
        """Get total priority sum"""
        return self.tree[0]
        
    def get_state(self) -> Dict:
        """Get tree state for checkpointing"""
        return {
            'tree': self.tree.copy(),
            'data': self.data.copy(),
            'write': self.write,
            'n_entries': self.n_entries,
            'max_recorded': self.max_recorded,
            'min_recorded': self.min_recorded
        }
        
    def set_state(self, state: Dict):
        """Set tree state from checkpoint"""
        self.tree = state['tree'].copy()
        self.data = state['data'].copy()
        self.write = state['write']
        self.n_entries = state['n_entries']
        self.max_recorded = state['max_recorded']
        self.min_recorded = state['min_recorded']
        
    @property
    def priorities(self) -> np.ndarray:
        """Get array of all priorities"""
        return self.tree[-self.capacity:]

class GRPOMoE(nn.Module):
    """
    GRPO-driven Mixture of Experts implementation
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.config = ConfigRegistry.get_config()
        # Override config with kwargs
        for k, v in kwargs.items():
            setattr(self.config, k, v)
        
        self.setup_experts()
        
    def setup_experts(self):
        # Expert networks
        self.experts = nn.ModuleList([
            Expert(self.config.d_model, self.config.d_ff, self.config.n_layers, self.config.dropout) for _ in range(self.config.num_experts)
        ])
        
        # Policy network for expert selection
        self.policy = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.LayerNorm(self.config.d_model // 2),
            nn.GELU(),
            nn.Linear(self.config.d_model // 2, self.config.num_experts)
        )
        
        # Value network for advantage estimation
        self.value = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.LayerNorm(self.config.d_model // 2),
            nn.GELU(),
            nn.Linear(self.config.d_model // 2, 1)
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 2
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with Gumbel-Top-k expert selection
        """
        batch_size = x.size(0)
        
        # Get expert logits
        logits = self.policy(x) / temperature
        
        # Gumbel-Top-k sampling
        gumbel = -torch.log(-torch.log(
            torch.rand_like(logits) + 1e-10) + 1e-10
        )
        scores = logits + gumbel
        
        # Select top-k experts
        topk_scores, topk_indices = scores.topk(top_k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)
        
        # Compute outputs from selected experts
        outputs = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # Mask for batches that selected this expert
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                expert_out, _ = expert(x[mask])
                outputs[mask] += expert_out
                
        # Compute value estimate
        value = self.value(x)
        
        # Return output and auxiliary info for GRPO loss
        return outputs, {
            'logits': logits,
            'selected_experts': topk_indices,
            'expert_weights': topk_weights,
            'value': value
        }
        
    def compute_grpo_loss(
        self,
        old_logits: torch.Tensor,
        new_logits: torch.Tensor,
        advantages: torch.Tensor,
        expert_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GRPO loss with PPO-style clipping
        L_RL = E_t[min(π(φ)/π_old A_t, clip(π/π_old, 0.8, 1.2)A_t)]
        """
        # Compute probability ratios
        old_probs = F.softmax(old_logits, dim=-1)
        new_probs = F.softmax(new_logits, dim=-1)
        
        # Mask for selected experts
        expert_probs_old = torch.gather(old_probs, 1, expert_mask)
        expert_probs_new = torch.gather(new_probs, 1, expert_mask)
        
        # Compute ratio π(φ)/π_old
        ratio = expert_probs_new / (expert_probs_old + 1e-8)
        
        # Clipped objective
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.config.clip_ratio,
            1 + self.config.clip_ratio
        )
        
        # Compute losses
        loss1 = ratio * advantages
        loss2 = clipped_ratio * advantages
        
        # Take minimum
        loss = -torch.min(loss1, loss2).mean()
        
        return loss