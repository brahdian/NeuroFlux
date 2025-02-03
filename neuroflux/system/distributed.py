import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Optional, List, Tuple, Union, Any
import os
import logging
from dataclasses import dataclass
import numpy as np
from contextlib import contextmanager
from pathlib import Path
import time
from datetime import datetime, timedelta

from ..utils.config_registry import ConfigRegistry

@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    world_size: int = int(os.environ.get("WORLD_SIZE", 1))
    rank: int = int(os.environ.get("RANK", 0))
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
    master_addr: str = os.environ.get("MASTER_ADDR", "localhost")
    master_port: str = os.environ.get("MASTER_PORT", "12355")
    backend: str = "nccl"
    timeout: float = 1800.0  # 30 minutes
    grad_sync_period: int = 16
    find_unused_parameters: bool = False

class DistributedTrainer:
    """
    Distributed trainer with improved dependency management
    """
    def __init__(self, model: torch.nn.Module, **kwargs):
        self.config = ConfigRegistry.get_config()
        # Override config with kwargs
        for k, v in kwargs.items():
            setattr(self.config, k, v)
            
        self.model = self._setup_model(model)
        self._monitor = None
        self._setup_distributed()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Initialize gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Gradient accumulation state
        self.grad_sync_counter = 0
        self.accumulated_grads = {}
        
        # Monitoring
        self.sync_times: List[float] = []
        self.compute_times: List[float] = []
        
    @property
    def monitor(self):
        """Lazy load monitor to avoid circular imports"""
        if self._monitor is None:
            from ..utils.monitoring import PerformanceMonitor
            self._monitor = PerformanceMonitor()
        return self._monitor
    
    def _setup_model(self, model: torch.nn.Module) -> DDP:
        """Setup distributed model"""
        # Wrap model in DDP
        model = DDP(
            model.to(self.device),
            device_ids=[self.config.local_rank],
            output_device=self.config.local_rank,
            find_unused_parameters=self.config.find_unused_parameters,
            gradient_as_bucket_view=True  # Memory optimization
        )
        return model
    
    def _setup_distributed(self):
        """Initialize distributed environment"""
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = self.config.master_port
        
        # Initialize process group
        dist.init_process_group(
            backend=self.config.backend,
            rank=self.config.rank,
            world_size=self.config.world_size,
            timeout=datetime.timedelta(seconds=self.config.timeout)
        )
        
        # Set device
        self.device = torch.device(f"cuda:{self.config.local_rank}")
        torch.cuda.set_device(self.device)
        
        # Enable cuDNN benchmarking
        torch.backends.cudnn.benchmark = True
        
    def _setup_optimizer(self):
        """Setup optimizer"""
        # Implementation of _setup_optimizer method
        pass
    
    def _setup_scheduler(self):
        """Setup scheduler"""
        # Implementation of _setup_scheduler method
        pass
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        grad_acc_steps: int
    ) -> Dict[str, float]:
        """Execute distributed training step"""
        start_time = time.time()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = self.model(
                batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device)
            )
            loss = outputs.loss / grad_acc_steps
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Accumulate gradients
        if self.grad_sync_counter == 0:
            self._accumulate_gradients()
        
        self.grad_sync_counter += 1
        
        metrics = {"loss": loss.item() * grad_acc_steps}
        
        # Synchronize gradients if needed
        if self.grad_sync_counter >= grad_acc_steps:
            sync_start = time.time()
            self._synchronize_gradients()
            self.sync_times.append(time.time() - sync_start)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            if self.scheduler is not None:
                self.scheduler.step()
                metrics["lr"] = self.scheduler.get_last_lr()[0]
            
            self.grad_sync_counter = 0
            
        self.compute_times.append(time.time() - start_time)
        
        return metrics
    
    def _accumulate_gradients(self):
        """Accumulate gradients for later synchronization"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.accumulated_grads:
                    self.accumulated_grads[name] = param.grad.clone()
                else:
                    self.accumulated_grads[name] += param.grad
    
    def _synchronize_gradients(self):
        """Synchronize gradients across nodes"""
        # All-reduce accumulated gradients
        for name, grad in self.accumulated_grads.items():
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad /= self.config.world_size
            
            # Update model gradients
            param = dict(self.model.named_parameters())[name]
            param.grad = grad.clone()
        
        self.accumulated_grads.clear()
    
    @contextmanager
    def sync_context(self):
        """Context manager for synchronized operations"""
        if dist.is_initialized():
            dist.barrier()
        try:
            yield
        finally:
            if dist.is_initialized():
                dist.barrier()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get training performance metrics"""
        metrics = {
            "avg_compute_time": np.mean(self.compute_times),
            "avg_sync_time": np.mean(self.sync_times) if self.sync_times else 0,
            "compute_std": np.std(self.compute_times),
            "sync_std": np.std(self.sync_times) if self.sync_times else 0,
            "grad_sync_efficiency": len(self.sync_times) / len(self.compute_times)
        }
        
        # Clear histories periodically
        if len(self.compute_times) > 1000:
            self.compute_times = self.compute_times[-1000:]
            self.sync_times = self.sync_times[-1000:]
            
        return metrics
    
    def save_checkpoint(
        self,
        path: str,
        metadata: Optional[Dict] = None
    ):
        """Save distributed checkpoint"""
        if self.config.rank == 0:  # Save only on master node
            state = {
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scaler': self.scaler.state_dict(),
                'metadata': metadata or {}
            }
            if self.scheduler is not None:
                state['scheduler'] = self.scheduler.state_dict()
                
            torch.save(state, path)
    
    def load_checkpoint(self, path: str) -> Optional[Dict]:
        """Load distributed checkpoint"""
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.config.local_rank}
        state = torch.load(path, map_location=map_location)
        
        self.model.module.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scaler.load_state_dict(state['scaler'])
        
        if self.scheduler is not None and 'scheduler' in state:
            self.scheduler.load_state_dict(state['scheduler'])
            
        return state.get('metadata')
    
    def cleanup(self):
        """Cleanup distributed training"""
        if dist.is_initialized():
            dist.destroy_process_group()

def launch_distributed(
    train_func,
    world_size: int,
    config: DistributedConfig
):
    """Launch distributed training across nodes"""
    if world_size > 1:
        torch.multiprocessing.spawn(
            _distributed_worker,
            args=(train_func, config),
            nprocs=world_size,
            join=True
        )
    else:
        train_func(config)

def _distributed_worker(
    local_rank: int,
    train_func,
    config: DistributedConfig
):
    """Worker function for distributed training"""
    # Update config with local rank
    config.local_rank = local_rank
    
    # Initialize process group
    dist.init_process_group(
        backend=config.backend,
        init_method=f"tcp://{config.master_addr}:{config.master_port}",
        world_size=config.world_size,
        rank=config.rank * config.world_size + local_rank
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    try:
        train_func(config)
    finally:
        dist.destroy_process_group() 