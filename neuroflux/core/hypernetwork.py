# neuroflux/hypernetworks.py
import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

class DifferentiableHyperNetwork(nn.Module):
    """
    Complete implementation of the differentiable hypernetwork from Section 3.1
    Learns SSM discretization (Δ), entropy coefficient (λ_H), and MoE gating temperature
    """
    def __init__(self, d_model: int, trust_region: bool = True):
        super().__init__()
        self.d_model = d_model
        self.trust_region = trust_region
        
        # Core feature extractor
        self.core = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU()
        )
        
        # Parameter-specific heads
        self.delta_net = ParameterHead(
            d_model // 4,
            bounds=(0.1, 2.0),
            init_value=1.0,
            name='delta'
        )
        
        self.lambda_net = ParameterHead(
            d_model // 4,
            bounds=(0.01, 0.99),
            init_value=0.5,
            name='lambda'
        )
        
        self.moe_temp_net = ParameterHead(
            d_model // 4,
            bounds=(0.5, 1.5),
            init_value=1.0,
            name='moe_temp'
        )
        
        # Trust region tracking
        if trust_region:
            self.register_buffer('param_history', torch.zeros(3, 100))  # Last 100 values
            self.register_buffer('trust_radii', torch.ones(3))  # One per parameter
            self.register_buffer('moving_averages', torch.zeros(3))
            self.register_buffer('moving_stds', torch.ones(3))
        
        # Phase-specific settings
        self.phase_configs = {
            'exploration': {
                'delta': {'center': 1.0, 'radius': 0.5},
                'lambda': {'center': 0.5, 'radius': 0.3},
                'moe_temp': {'center': 1.0, 'radius': 0.3}
            },
            'exploitation': {
                'delta': {'center': 0.8, 'radius': 0.3},
                'lambda': {'center': 0.3, 'radius': 0.2},
                'moe_temp': {'center': 0.8, 'radius': 0.2}
            },
            'consolidation': {
                'delta': {'center': 0.5, 'radius': 0.2},
                'lambda': {'center': 0.1, 'radius': 0.1},
                'moe_temp': {'center': 0.5, 'radius': 0.1}
            }
        }
        
    def forward(
        self, 
        h_t: torch.Tensor,
        phase: str = 'exploitation',
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with trust region constraints and phase-specific adjustments
        
        Args:
            h_t: Hidden state tensor (batch_size, d_model)
            phase: Current training phase
            return_stats: Whether to return trust region statistics
        """
        # Get phase-specific settings
        phase_config = self.phase_configs[phase]
        
        # Extract features
        features = self.core(h_t)
        
        # Compute parameters with trust region constraints if enabled
        if self.trust_region:
            delta = self.delta_net(
                features,
                self.trust_radii[0],
                phase_config['delta']
            )
            lambda_h = self.lambda_net(
                features,
                self.trust_radii[1],
                phase_config['lambda']
            )
            moe_temp = self.moe_temp_net(
                features,
                self.trust_radii[2],
                phase_config['moe_temp']
            )
            
            # Update trust regions
            self._update_trust_regions(
                torch.stack([delta.mean(), lambda_h.mean(), moe_temp.mean()])
            )
        else:
            # Direct parameter prediction without trust regions
            delta = self.delta_net(features)
            lambda_h = self.lambda_net(features)
            moe_temp = self.moe_temp_net(features)
        
        if return_stats and self.trust_region:
            return delta, lambda_h, moe_temp, {
                'trust_radii': self.trust_radii.clone(),
                'moving_averages': self.moving_averages.clone(),
                'moving_stds': self.moving_stds.clone()
            }
        
        return delta, lambda_h, moe_temp
    
    def _update_trust_regions(self, new_params: torch.Tensor):
        """Update trust regions based on parameter stability"""
        # Update parameter history
        self.param_history = torch.roll(self.param_history, -1, dims=1)
        self.param_history[:, -1] = new_params
        
        # Update moving statistics
        decay = 0.99
        self.moving_averages = (
            decay * self.moving_averages + 
            (1 - decay) * new_params
        )
        
        # Update moving standard deviations
        diff = new_params - self.moving_averages
        self.moving_stds = torch.sqrt(
            decay * self.moving_stds**2 +
            (1 - decay) * diff**2
        )
        
        # Compute stability metrics
        stability = 1.0 - (
            self.moving_stds / 
            (self.moving_averages.abs() + 1e-6)
        ).clamp(0, 1)
        
        # Update trust radii with momentum
        momentum = 0.9
        self.trust_radii = (
            momentum * self.trust_radii +
            (1 - momentum) * stability
        )

class ParameterHead(nn.Module):
    """
    Parameter-specific prediction head with bounds and trust region enforcement
    """
    def __init__(
        self,
        input_dim: int,
        bounds: Tuple[float, float],
        init_value: float,
        name: str
    ):
        super().__init__()
        self.bounds = bounds
        self.name = name
        
        # Initialize network with bias to match init_value
        self.net = nn.Linear(input_dim, 1)
        with torch.no_grad():
            self.net.bias.fill_(
                math.log(
                    (init_value - bounds[0]) /
                    (bounds[1] - init_value)
                )
            )
    
    def forward(
        self,
        x: torch.Tensor,
        trust_radius: Optional[torch.Tensor] = None,
        phase_config: Optional[dict] = None
    ) -> torch.Tensor:
        """Forward pass with optional trust region enforcement"""
        raw_output = self.net(x)
        
        if trust_radius is not None and phase_config is not None:
            # Apply trust region constraint
            center = phase_config['center']
            radius = phase_config['radius'] * trust_radius
            
            # Clamp output to trust region
            bounded_output = torch.clamp(
                raw_output,
                center - radius,
                center + radius
            )
        else:
            bounded_output = raw_output
        
        # Scale to parameter bounds
        min_val, max_val = self.bounds
        scaled_output = min_val + (max_val - min_val) * torch.sigmoid(bounded_output)
        
        return scaled_output

def init_hypernetwork(model: nn.Module) -> DifferentiableHyperNetwork:
    """Initialize hypernetwork with model's hidden dimension"""
    d_model = model.config.hidden_size
    return DifferentiableHyperNetwork(d_model)