"""
NeuroFlux 600M Model Training on Colab TPU
=========================================

Initial training with 10GB dataset and scaling capabilities.
"""

import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import GPT2Tokenizer
import wandb
from tqdm import tqdm
from typing import Dict, List, Any
import numpy as np

from neuroflux.core.model import SSMXLSTMFusion
from neuroflux.scaling.scale_manager import NeuroFluxScaler
from neuroflux.scaling.state_manager import StateManager
from neuroflux.utils.config import ConfigRegistry

class TPUConfig:
    """Configuration for 10GB dataset training run"""
    # Model Architecture (600M params)
    D_MODEL = 1024
    N_LAYERS = 24
    N_HEADS = 16
    
    # Training
    BATCH_SIZE = 16  # Reduced batch size for 10GB dataset
    TOTAL_STEPS = 50_000  # Reduced steps for initial validation
    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 1000
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # Scaling
    SCALE_FACTOR = 1.5
    MIN_SCALE_STEPS = 5000
    SCALE_METRIC_THRESHOLD = 0.1
    
    # System
    LOG_FREQ = 10
    CHECKPOINT_FREQ = 500
    MAX_DATASET_SIZE_GB = 10
    
    # TPU specific
    NUM_CORES = 8
    DEVICE = 'xla'

class TPUTrainer:
    """TPU-optimized trainer with scaling capabilities"""
    
    def __init__(self, 
                 model: SSMXLSTMFusion,
                 config: TPUConfig,
                 train_loader: DataLoader,
                 device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Wrap dataloader for TPU
        self.train_loader = pl.MpDeviceLoader(train_loader, device)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE
        )
        
        # Initialize scaling components
        self.scaler = NeuroFluxScaler(model)
        self.state_manager = StateManager()
        
        # Metrics tracking for scaling decisions
        self.loss_history = []
        self.steps_since_scale = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch['input_ids'])
        loss = outputs.loss / self.config.GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (self.step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
            self.optimizer.step()
            xm.mark_step()
            
        return loss.item()

    def should_scale(self, current_loss: float) -> bool:
        """Determine if model should be scaled up"""
        self.loss_history.append(current_loss)
        self.steps_since_scale += 1
        
        if self.steps_since_scale < self.config.MIN_SCALE_STEPS:
            return False
            
        # Check if loss has plateaued
        if len(self.loss_history) >= 100:
            recent_loss = np.mean(self.loss_history[-100:])
            older_loss = np.mean(self.loss_history[-200:-100])
            relative_improvement = (older_loss - recent_loss) / older_loss
            
            return relative_improvement < self.config.SCALE_METRIC_THRESHOLD
            
        return False

    def scale_model(self):
        """Scale up model size"""
        xm.master_print("Scaling up model...")
        self.model = self.scaler.scale_up(
            scale_factor=self.config.SCALE_FACTOR,
            preserve_weights=True
        )
        
        # Reset optimizer with new parameters
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE
        )
        
        self.steps_since_scale = 0
        xm.master_print(f"Model scaled up. New parameter count: {self.scaler.get_param_count():,}")

    def save_checkpoint(self, step: int, loss: float, best_loss: float):
        """Save training checkpoint"""
        if loss < best_loss:
            state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'step': step,
                'loss': loss,
                'config': self.config,
                'scaler_state': self.scaler.get_state(),
                'loss_history': self.loss_history
            }
            path = f'checkpoints/model_step_{step}_loss_{loss:.4f}.pt'
            xm.save(state, path)
            return loss
        return best_loss

    def train(self):
        """TPU training loop with scaling"""
        self.step = 0
        best_loss = float('inf')
        
        while self.step < self.config.TOTAL_STEPS:
            for batch in self.train_loader:
                loss = self.train_step(batch)
                
                if self.step % self.config.LOG_FREQ == 0:
                    xm.master_print(f"Step {self.step}: Loss = {loss:.4f}")
                    if xm.is_master_ordinal():
                        wandb.log({
                            'step': self.step,
                            'loss': loss,
                            'lr': self.optimizer.param_groups[0]['lr'],
                            'model_size': self.scaler.get_param_count()
                        })
                
                if self.step % self.config.CHECKPOINT_FREQ == 0 and xm.is_master_ordinal():
                    best_loss = self.save_checkpoint(self.step, loss, best_loss)
                
                # Check scaling conditions
                if self.should_scale(loss):
                    self.scale_model()
                
                self.step += 1
                if self.step >= self.config.TOTAL_STEPS:
                    break
                
            # TPU optimization barrier
            xm.mark_step()

def load_10gb_dataset():
    """Load and prepare 10GB training dataset"""
    dataset = load_dataset(
        'text',
        data_files={'train': 'path/to/10gb/data.txt'},
        split='train'
    )
    
    # Trim to 10GB if larger
    if dataset.size_in_bytes > 10 * 1024 * 1024 * 1024:  # 10GB in bytes
        num_samples = int(len(dataset) * (10 * 1024 * 1024 * 1024 / dataset.size_in_bytes))
        dataset = dataset.select(range(num_samples))
    
    return dataset

def main():
    """Main training function"""
    # Initialize wandb
    if xm.is_master_ordinal():
        wandb.init(project="neuroflux-10gb-tpu")
    
    # Load dataset
    dataset = load_10gb_dataset()
    train_loader = DataLoader(
        dataset,
        batch_size=TPUConfig.BATCH_SIZE,
        shuffle=True
    )
    
    # Initialize model and trainer
    device = xm.xla_device()
    model = SSMXLSTMFusion(config=TPUConfig)
    trainer = TPUTrainer(model, TPUConfig, train_loader, device)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    xmp.spawn(main, nprocs=TPUConfig.NUM_CORES) 