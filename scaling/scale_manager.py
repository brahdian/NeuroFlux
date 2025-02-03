"""
NeuroFlux Scaling Manager
========================

Adds progressive scaling capabilities to existing NeuroFlux models.
Works with the current SSM-XLSTM-MoE architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math
import logging
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from ..core.model import SSMXLSTMFusion
from ..core.controllers import GRPOMoE
from ..utils.config_registry import ConfigRegistry
from ..system.raid import RAIDMemory

@dataclass
class ScaleStep:
    """Defines a single scaling step"""
    name: str
    d_model: int
    n_layers: int
    n_experts: int
    xlstm_scales: int
    target_params: float  # in billions
    
    def __post_init__(self):
        self.actual_params = self.calculate_params()
    
    def calculate_params(self) -> float:
        """Calculate actual parameter count"""
        expert_params = self.d_model * self.d_model * 4 * self.n_experts
        layer_params = self.d_model * self.d_model * 4 * self.n_layers
        xlstm_params = self.d_model * self.xlstm_scales * 4
        return (expert_params + layer_params + xlstm_params) / 1e9  # in billions

class NeuroFluxScaler:
    """Manages progressive scaling of NeuroFlux models"""
    
    SCALE_STEPS = [
        ScaleStep("600M", 1024, 24, 16, 4, 0.6),
        ScaleStep("6B", 1536, 32, 32, 5, 6.0),
        ScaleStep("60B", 2304, 42, 64, 6, 60.0),
        ScaleStep("600B", 3456, 56, 128, 7, 600.0)
    ]
    
    def __init__(self, 
                 base_model: Optional[SSMXLSTMFusion] = None,
                 checkpoint_dir: str = "checkpoints/scaling"):
        self.model = base_model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_step = 0
        self.logger = self._setup_logger()
        self.raid = RAIDMemory()
    
    def _setup_logger(self) -> logging.Logger:
        """Initialize scaling-specific logger"""
        logger = logging.getLogger("NeuroFluxScaler")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        return logger

    def scale_up(self, preserve_knowledge: bool = True) -> SSMXLSTMFusion:
        """Scale model to next size while preserving knowledge"""
        if self.current_step >= len(self.SCALE_STEPS) - 1:
            raise ValueError("Already at maximum scale")
            
        current_scale = self.SCALE_STEPS[self.current_step]
        next_scale = self.SCALE_STEPS[self.current_step + 1]
        
        self.logger.info(f"Scaling from {current_scale.name} to {next_scale.name}")
        
        # Create new model with scaled dimensions
        new_config = self._create_scaled_config(next_scale)
        new_model = SSMXLSTMFusion(config=new_config)
        
        if preserve_knowledge:
            new_model = self._transfer_knowledge(
                source_model=self.model,
                target_model=new_model,
                current_scale=current_scale,
                next_scale=next_scale
            )
        
        self.model = new_model
        self.current_step += 1
        return self.model
    
    def _create_scaled_config(self, scale_step: ScaleStep) -> Dict:
        """Create configuration for scaled model"""
        config = ConfigRegistry.get_config()
        
        # Update architecture parameters
        config.D_MODEL = scale_step.d_model
        config.N_LAYERS = scale_step.n_layers
        config.N_EXPERTS = scale_step.n_experts
        config.XLSTM_SCALES = scale_step.xlstm_scales
        
        # Scale learning rate and batch size
        config.LEARNING_RATE *= math.sqrt(scale_step.d_model / 1024)
        config.BATCH_SIZE = max(32, config.BATCH_SIZE // 2)
        
        return config
    
    def _transfer_knowledge(self, 
                          source_model: SSMXLSTMFusion,
                          target_model: SSMXLSTMFusion,
                          current_scale: ScaleStep,
                          next_scale: ScaleStep) -> SSMXLSTMFusion:
        """Transfer knowledge between models of different scales"""
        
        # 1. Transfer SSM parameters
        self._transfer_ssm_knowledge(
            source_model.ssm,
            target_model.ssm,
            current_scale,
            next_scale
        )
        
        # 2. Transfer XLSTM states
        self._transfer_xlstm_knowledge(
            source_model.xlstm,
            target_model.xlstm,
            current_scale,
            next_scale
        )
        
        # 3. Transfer and expand experts
        self._transfer_expert_knowledge(
            source_model.controller,
            target_model.controller,
            current_scale,
            next_scale
        )
        
        return target_model
    
    def save_checkpoint(self, step: int, metrics: Dict = None):
        """Save scaling checkpoint with RAID protection"""
        checkpoint_path = self.checkpoint_dir / f"scale_{self.SCALE_STEPS[self.current_step].name}_step_{step}.pt"
        
        checkpoint = {
            'model_state': self.model.state_dict(),
            'scale_step': self.current_step,
            'config': self.model.config,
            'metrics': metrics,
            'scale_history': [s.name for s in self.SCALE_STEPS[:self.current_step + 1]]
        }
        
        # Save with RAID protection
        self.raid.save_checkpoint(checkpoint, checkpoint_path)
        self.logger.info(f"Saved scaling checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, path: str) -> Tuple[SSMXLSTMFusion, int]:
        """Load checkpoint and return model and scale step"""
        checkpoint = self.raid.load_checkpoint(path)
        
        # Restore model and scaling state
        self.current_step = checkpoint['scale_step']
        config = checkpoint['config']
        
        self.model = SSMXLSTMFusion(config=config)
        self.model.load_state_dict(checkpoint['model_state'])
        
        self.logger.info(f"Loaded model at scale: {self.SCALE_STEPS[self.current_step].name}")
        return self.model, self.current_step

# Example usage
if __name__ == "__main__":
    # Initialize base 600M model
    base_model = SSMXLSTMFusion(config=ConfigRegistry.get_config())
    
    # Create scaler
    scaler = NeuroFluxScaler(base_model)
    
    # Training loop with progressive scaling
    for epoch in range(10):
        # Train current scale
        # ... training code ...
        
        # Save checkpoint
        scaler.save_checkpoint(
            step=epoch,
            metrics={'loss': current_loss, 'accuracy': current_acc}
        )
        
        # Scale up every few epochs
        if epoch > 0 and epoch % 3 == 0:
            try:
                model = scaler.scale_up(preserve_knowledge=True)
                print(f"Scaled up to {scaler.SCALE_STEPS[scaler.current_step].name}")
            except ValueError:
                print("Reached maximum scale")
                break