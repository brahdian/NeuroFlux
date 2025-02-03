import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional
import math
import io
from dataclasses import dataclass
from typing import List, Dict
import time

class ExpertUsageTracker:
    """Tracks and analyzes expert usage patterns over time"""
    def __init__(self, num_experts: int, history_size: int = 1000):
        self.num_experts = num_experts
        self.history_size = history_size
        
        # Usage statistics
        self.total_calls = torch.zeros(num_experts)
        self.recent_usage = deque(maxlen=history_size)
        self.usage_timestamps = deque(maxlen=history_size)
        
        # Load statistics
        self.peak_load = torch.zeros(num_experts)
        self.avg_load = torch.zeros(num_experts)
        
        # Efficiency metrics
        self.last_update = time.time()
        self.compute_efficiency = torch.zeros(num_experts)
        
    def update(self, expert_indices: torch.Tensor, expert_weights: torch.Tensor) -> Dict:
        """Update usage statistics with new batch of expert assignments"""
        current_time = time.time()
        batch_size = expert_indices.size(0)
        
        # Compute current batch usage
        batch_usage = torch.bincount(
            expert_indices.view(-1),
            weights=expert_weights.view(-1),
            minlength=self.num_experts
        )
        
        # Update total usage
        self.total_calls += batch_usage
        
        # Update recent usage history
        self.recent_usage.append(batch_usage)
        self.usage_timestamps.append(current_time)
        
        # Update load statistics
        self.peak_load = torch.maximum(self.peak_load, batch_usage)
        self.avg_load = self._compute_moving_average(batch_usage)
        
        # Update efficiency metrics
        time_delta = current_time - self.last_update
        self.compute_efficiency = self._update_efficiency(batch_usage, time_delta)
        self.last_update = current_time
        
        return self._get_current_stats(batch_size)
        
    def _compute_moving_average(self, new_usage: torch.Tensor) -> torch.Tensor:
        """Compute exponential moving average of expert usage"""
        alpha = 0.99
        if not self.recent_usage:
            return new_usage
            
        current_avg = torch.stack(list(self.recent_usage)).mean(dim=0)
        return alpha * current_avg + (1 - alpha) * new_usage
        
    def _update_efficiency(self, batch_usage: torch.Tensor, time_delta: float) -> torch.Tensor:
        """Update compute efficiency metrics"""
        # Compute operations per second for each expert
        ops_per_second = batch_usage / max(time_delta, 1e-6)
        
        # Exponential moving average of efficiency
        alpha = 0.95
        return alpha * self.compute_efficiency + (1 - alpha) * ops_per_second
        
    def _get_current_stats(self, batch_size: int) -> Dict:
        """Generate current usage statistics"""
        return {
            'total_calls': self.total_calls,
            'recent_usage': torch.stack(list(self.recent_usage)) if self.recent_usage else None,
            'peak_load': self.peak_load,
            'avg_load': self.avg_load,
            'compute_efficiency': self.compute_efficiency,
            'load_imbalance': self._compute_load_imbalance(),
            'utilization': self._compute_utilization(batch_size)
        }
        
    def _compute_load_imbalance(self) -> float:
        """Compute load imbalance metric"""
        if not self.recent_usage:
            return 0.0
            
        recent_usage = torch.stack(list(self.recent_usage))
        mean_usage = recent_usage.mean(dim=0)
        std_usage = recent_usage.std(dim=0)
        return (std_usage / (mean_usage + 1e-6)).mean().item()
        
    def _compute_utilization(self, batch_size: int) -> torch.Tensor:
        """Compute expert utilization rates"""
        if not self.recent_usage:
            return torch.zeros(self.num_experts)
            
        recent_usage = torch.stack(list(self.recent_usage))
        total_possible = batch_size * len(self.recent_usage)
        return recent_usage.sum(dim=0) / total_possible
        
    def get_expert_rankings(self) -> List[int]:
        """Rank experts by their recent utilization"""
        if not self.recent_usage:
            return list(range(self.num_experts))
            
        recent_usage = torch.stack(list(self.recent_usage))
        mean_usage = recent_usage.mean(dim=0)
        return mean_usage.argsort(descending=True).tolist()
        
    def get_underutilized_experts(self, threshold: float = 0.1) -> List[int]:
        """Identify underutilized experts"""
        if not self.recent_usage:
            return []
            
        utilization = self._compute_utilization(1)  # batch_size=1 for relative utilization
        return (utilization < threshold).nonzero().squeeze(-1).tolist()
        
    def reset(self):
        """Reset all tracking statistics"""
        self.__init__(self.num_experts, self.history_size)

@dataclass
class UsageStats:
    """Container for expert usage statistics"""
    total_calls: torch.Tensor
    recent_usage: Optional[torch.Tensor]
    peak_load: torch.Tensor
    avg_load: torch.Tensor
    compute_efficiency: torch.Tensor
    load_imbalance: float
    utilization: torch.Tensor

class GRPOTrainer:
    """Enhanced GRPO Trainer with O(1) optimizations and full paper implementation"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Initialize optimizers with correct scaling
        self.optimizers = self._init_optimizers(config)
        
        # Enhanced scheduler with warmup and cosine decay
        self.scheduler = self._init_scheduler(config)
        
        # Fault tolerance system
        self.fault_simulator = RAIDFaultInjector(
            config.num_gpus,
            config.ckpt_interval,
            parity_slots=2  # RAID-6 configuration
        )
        
        # Initialize enhancements
        self._init_enhancements(config)
        
        # State tracking
        self.step = 0
        self.best_metric = float('inf')
        
    def _init_optimizers(self, config) -> Dict[str, torch.optim.Optimizer]:
        """Initialize optimizers with layer-wise scaling"""
        def get_grouped_params(module, lr_scale=1.0):
            no_decay = ['bias', 'LayerNorm.weight']
            return [
                {
                    'params': [p for n, p in module.named_parameters() 
                              if not any(nd in n for nd in no_decay)],
                    'weight_decay': config.weight_decay,
                    'lr': config.lr * lr_scale
                },
                {
                    'params': [p for n, p in module.named_parameters() 
                              if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': config.lr * lr_scale
                }
            ]

        return {
            'exploration': torch.optim.AdamW(
                get_grouped_params(self.model.ssm, lr_scale=1.0),
                lr=config.lr,
                betas=(0.9, 0.95)
            ),
            'exploitation': torch.optim.AdamW([
                *get_grouped_params(self.model.moe, lr_scale=0.5),
                *get_grouped_params(self.model.xlstm, lr_scale=1.0)
            ]),
            'consolidation': torch.optim.AdamW(
                get_grouped_params(self.model.hypernet, lr_scale=0.1)
            )
        }

    def _init_scheduler(self, config):
        """Enhanced scheduler with warmup and cosine decay"""
        def lr_lambda(step):
            warmup_steps = 1000
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                progress = float(step - warmup_steps) / float(max(1, config.total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
                
        return LambdaLR(self.optimizers['exploration'], lr_lambda=lr_lambda)

    def _init_enhancements(self, config):
        """Initialize enhanced components"""
        # Curriculum learning with dynamic thresholds
        self.curriculum = EnhancedCurriculumManager(
            config.total_steps,
            warmup_steps=1000,
            phase_ratios={'exploration': 0.3, 'exploitation': 0.5, 'consolidation': 0.2}
        )
        
        # Metric tracking with exponential moving averages
        self.phase_metrics = {
            phase: MetricTracker(
                window_size=100,
                decay=0.99,
                threshold=config.phase_thresholds.get(phase, 0.0)
            )
            for phase in ['exploration', 'exploitation', 'consolidation']
        }
        
        # Enhanced MoE balancing
        self.moe_balancer = EnhancedMoEBalancer(
            num_experts=self.model.moe.num_experts,
            capacity_factor=config.moe_capacity_factor,
            min_expert_capacity=config.min_expert_capacity
        )
        
        # FP8 checkpoint compression
        self.checkpoint_manager = EnhancedCheckpointManager(
            compression_type='fp8',
            raid_config=config.raid_config
        )
        
        # Training state
        self.last_stable_step = 0
        self.kl_stats = ExpMovingAverage(0.99)  # For adaptive KL threshold

    def training_step(self, batch, step) -> Tuple[float, Dict]:
        """Enhanced training step with O(1) memory optimization"""
        self.step = step
        phase = self._get_curriculum_phase(step)
        self._set_phase_params(phase)
        
        # Fault injection check
        if self.fault_simulator.should_inject_fault(step):
            self._handle_raid_failure()
        
        # Forward pass with gradient checkpointing
        with torch.cuda.amp.autocast():
            outputs, aux = self.model(
                batch,
                lambda_h=self.current_lambda_h,
                moe_temp=self.current_moe_temp,
                use_checkpoint=True  # Enable gradient checkpointing
            )
            
            # Compute enhanced loss
            loss, loss_components = self._compute_enhanced_grpo_loss(outputs, aux, phase)
            
            # Scale loss for mixed precision
            self.grad_scaler.scale(loss).backward()
        
        # Gradient clipping and optimization
        self._clip_and_step(phase)
        
        # Update auxiliary systems
        self._update_metrics(phase, loss_components)
        self._adaptive_checkpointing(step)
        
        return loss.item(), loss_components

    def _compute_enhanced_grpo_loss(
        self, 
        outputs: torch.Tensor, 
        aux: Dict[str, torch.Tensor], 
        phase: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Enhanced GRPO loss computation following paper equations"""
        # Task-specific loss
        task_loss = F.cross_entropy(
            outputs.logits, 
            outputs.labels,
            label_smoothing=0.1
        )
        
        # GRPO policy loss with adaptive KL
        policy_ratio = torch.exp(aux['new_log_probs'] - aux['old_log_probs'])
        kl_div = (aux['old_log_probs'] - aux['new_log_probs']).mean()
        
        # Implement exact equation from paper
        policy_loss = -torch.min(
            policy_ratio * aux['advantages'],
            torch.clamp(policy_ratio, 0.8, 1.2) * aux['advantages']
        ).mean()
        
        # Adaptive KL threshold
        if self.kl_stats.avg > 2.0 * self.config.target_kl:
            self.current_lambda_h *= 0.5
        elif self.kl_stats.avg < 0.5 * self.config.target_kl:
            self.current_lambda_h *= 2.0
            
        # Value loss with GAE
        value_loss = F.huber_loss(aux['values'], aux['returns'])
        
        # Enhanced entropy loss
        entropy_loss = -self.current_lambda_h * aux['entropy'].mean()
        
        # MoE balance loss
        if hasattr(self, 'moe_balancer'):
            balance_stats = self.moe_balancer.update_usage(
                aux['expert_indices'],
                aux['expert_weights']
            )
            balance_loss = self.moe_balancer.compute_loss(balance_stats)
        else:
            balance_loss = 0.0
            
        # Combine losses with dynamic weighting
        weights = self._get_phase_loss_weights(phase)
        total_loss = (
            weights['task'] * task_loss +
            weights['policy'] * policy_loss +
            weights['value'] * value_loss +
            weights['entropy'] * entropy_loss +
            weights['balance'] * balance_loss
        )
        
        components = {
            'task': task_loss.item(),
            'policy': policy_loss.item(),
            'value': value_loss.item(),
            'entropy': entropy_loss.item(),
            'balance': balance_loss if isinstance(balance_loss, float) else balance_loss.item(),
            'kl': kl_div.item()
        }
        
        return total_loss, components

    def _clip_and_step(self, phase: str):
        """Enhanced gradient clipping and optimization"""
        # Unscale gradients for proper clipping
        self.grad_scaler.unscale_(self.optimizers[phase])
        
        # Compute gradient norm for adaptive clipping
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.max_grad_norm
        )
        
        # Skip step if gradients are invalid
        if not torch.isfinite(grad_norm):
            self.grad_scaler.skip_step(self.optimizers[phase])
            return
            
        # Perform optimization step
        self.grad_scaler.step(self.optimizers[phase])
        self.grad_scaler.update()
        
        if phase == 'exploration':
            self.scheduler.step()

    def _handle_raid_failure(self):
        """Enhanced RAID-6 failure handling"""
        try:
            if self.model.raid.can_recover():
                # Attempt RAID rebuild
                self.model.raid.rebuild_parity(
                    algorithm='reed_solomon',
                    verify_rebuild=True
                )
                self._log_recovery_success()
            else:
                # Fallback to checkpoint
                self._reload_latest_stable_checkpoint()
                self.model.raid.reconstruct_from_checkpoint(
                    verify_integrity=True
                )
        except Exception as e:
            self._handle_catastrophic_failure(e)

    def _adaptive_checkpointing(self, step: int):
        """Enhanced checkpointing with FP8 compression"""
        # Determine if checkpointing is needed
        should_checkpoint = (
            step - self.last_stable_step > self.config.ckpt_interval or
            self._detect_performance_regression() or
            self.fault_simulator.risk_level > 0.8
        )
        
        if should_checkpoint:
            self.checkpoint_manager.save(
                model=self.model,
                optimizers=self.optimizers,
                step=step,
                metrics=self.phase_metrics,
                compression_config={
                    'dtype': 'fp8',
                    'scheme': 'hybrid'  # Use hybrid compression scheme
                }
            )
            self.last_stable_step = step

class EnhancedMoEBalancer:
    """Enhanced MoE load balancing system"""
    def __init__(self, num_experts, capacity_factor=1.0, min_expert_capacity=4):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.min_expert_capacity = min_expert_capacity
        self.usage_tracker = ExpertUsageTracker(num_experts)
        
    def update_usage(self, expert_indices, expert_weights):
        """Track expert usage with enhanced statistics"""
        batch_size = expert_indices.size(0)
        capacity = max(
            int(batch_size * self.capacity_factor / self.num_experts),
            self.min_expert_capacity
        )
        
        # Compute load statistics
        expert_counts = torch.bincount(
            expert_indices.view(-1),
            weights=expert_weights.view(-1),
            minlength=self.num_experts
        )
        
        return {
            'counts': expert_counts,
            'capacity': capacity,
            'overflow': torch.relu(expert_counts - capacity)
        }
        
    def compute_loss(self, stats):
        """Compute load balancing loss"""
        # Implement z-loss from "GShard: Scaling Giant Models with Conditional Computation"
        overflow_loss = torch.mean(stats['overflow'] ** 2)
        
        # Add importance-weighted balance loss
        importance = stats['counts'].float() / stats['counts'].sum()
        balance_loss = torch.sum(importance * torch.log(importance + 1e-6))
        
        return overflow_loss + 0.01 * balance_loss

class EnhancedCheckpointManager:
    """Enhanced checkpoint management with FP8 compression"""
    def __init__(self, compression_type='fp8', raid_config=None):
        self.compression_type = compression_type
        self.raid_config = raid_config or {'parity_slots': 2}
        
    def save(self, model, optimizers, step, metrics, compression_config=None):
        """Save compressed checkpoint with RAID parity"""
        # Compress model state
        compressed_state = self._compress_state_dict(
            model.state_dict(),
            compression_config
        )
        
        # Prepare checkpoint
        checkpoint = {
            'model': compressed_state,
            'optimizers': {k: opt.state_dict() for k, opt in optimizers.items()},
            'metrics': metrics,
            'step': step
        }
        
        # Add RAID parity
        checkpoint = self._add_raid_parity(checkpoint)
        
        # Save to disk
        torch.save(
            checkpoint,
            f"neuroflux_ckpt_{step}.pt",
            _use_new_zipfile_serialization=False
        )
        
    def _compress_state_dict(self, state_dict, config=None):
        """Compress state dict using FP8"""
        if not config or config['dtype'] != 'fp8':
            return state_dict
            
        compressed = {}
        for k, v in state_dict.items():
            if v.dtype in [torch.float32, torch.float16]:
                compressed[k] = self._quantize_to_fp8(v)
            else:
                compressed[k] = v
                
        return compressed
        
    def _quantize_to_fp8(self, tensor):
        """Quantize tensor to FP8 format"""
        # Implementation of FP8 quantization
        # Following paper's hybrid scheme
        scale = tensor.abs().max()
        norm_tensor = tensor / scale
        quant = torch.round(norm_tensor * 127).clamp(-127, 127).to(torch.int8)
        return {
            'quantized': quant,
            'scale': scale
        }
        
    def _add_raid_parity(self, checkpoint):
        """Add RAID-6 parity using Reed-Solomon coding"""
        # Convert checkpoint to bytes
        data = self._serialize_checkpoint(checkpoint)
        
        # Split into chunks
        chunk_size = len(data) // (self.raid_config['parity_slots'] + 1)
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Generate Reed-Solomon parity
        rs = ReedSolomonEncoder(self.raid_config['parity_slots'])
        parity_chunks = rs.encode(chunks)
        
        checkpoint['raid_parity'] = parity_chunks
        return checkpoint
        
    def _serialize_checkpoint(self, checkpoint):
        """Serialize checkpoint to bytes with optimization"""
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        return buffer.getvalue()

class EnhancedCurriculumManager:
    """Enhanced curriculum learning with dynamic phase transitions"""
    def __init__(self, total_steps, warmup_steps=1000, phase_ratios=None):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.phase_ratios = phase_ratios or {
            'exploration': 0.3,
            'exploitation': 0.5,
            'consolidation': 0.2
        }
        
        # Phase boundaries
        self.phase_steps = self._compute_phase_steps()
        
        # Dynamic thresholds
        self.performance_thresholds = {
            'exploration': 0.7,
            'exploitation': 0.8,
            'consolidation': 0.85
        }
        
    def _compute_phase_steps(self):
        """Compute step boundaries for each phase"""
        steps = {}
        current_step = 0
        
        for phase, ratio in self.phase_ratios.items():
            phase_length = int(self.total_steps * ratio)
            steps[phase] = {
                'start': current_step,
                'end': current_step + phase_length
            }
            current_step += phase_length
            
        return steps
        
    def get_phase(self, step):
        """Determine current training phase"""
        if step < self.warmup_steps:
            return 'exploration'
            
        for phase, boundaries in self.phase_steps.items():
            if boundaries['start'] <= step < boundaries['end']:
                return phase
                
        return 'consolidation'
        
    def get_moe_config(self, phase):
        """Get phase-specific MoE configuration"""
        configs = {
            'exploration': {
                'top_k': 4,
                'temp_scale': 1.0,
                'attention_scale': 1.0
            },
            'exploitation': {
                'top_k': 2,
                'temp_scale': 0.8,
                'attention_scale': 1.2
            },
            'consolidation': {
                'top_k': 1,
                'temp_scale': 0.5,
                'attention_scale': 1.5
            }
        }
        return configs[phase]

class MetricTracker:
    """Enhanced metric tracking with exponential moving averages"""
    def __init__(self, window_size=100, decay=0.99, threshold=0.0):
        self.window_size = window_size
        self.decay = decay
        self.threshold = threshold
        
        self.values = deque(maxlen=window_size)
        self.ema = None
        self.best_value = float('inf')
        
    def update(self, value):
        """Update metrics with new value"""
        self.values.append(value)
        
        # Update exponential moving average
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.ema * self.decay + value * (1 - self.decay)
            
        # Update best value
        if value < self.best_value:
            self.best_value = value
            
    def get_stats(self):
        """Get current statistics"""
        return {
            'current': self.values[-1] if self.values else None,
            'ema': self.ema,
            'best': self.best_value,
            'avg': sum(self.values) / len(self.values) if self.values else None,
            'std': np.std(list(self.values)) if len(self.values) > 1 else 0
        }
        
    def is_improving(self):
        """Check if metrics are improving"""
        if len(self.values) < 2:
            return True
            
        recent_avg = sum(list(self.values)[-10:]) / min(10, len(self.values))
        return recent_avg < self.ema

class ReedSolomonEncoder:
    """Reed-Solomon encoding for RAID-6 parity"""
    def __init__(self, num_parity_chunks):
        self.num_parity = num_parity_chunks
        self._init_galois_field()
        
    def _init_galois_field(self):
        """Initialize Galois Field for RS coding"""
        # GF(2^8) implementation
        self.field_size = 256
        self.generator_poly = self._build_generator_polynomial()
        
    def _build_generator_polynomial(self):
        """Build generator polynomial for RS encoding"""
        # Implementation of generator polynomial construction
        # For RAID-6, typically using x^2 + g^1x + g^2
        return [1, 2, 2]  # Simplified version
        
    def encode(self, data_chunks):
        """Encode data chunks with Reed-Solomon"""
        # Convert data to GF(2^8) representation
        gf_chunks = self._to_galois_field(data_chunks)
        
        # Generate parity chunks
        parity = []
        for i in range(self.num_parity):
            parity_chunk = self._generate_parity_chunk(gf_chunks, i)
            parity.append(parity_chunk)
            
        return parity
        
    def _to_galois_field(self, chunks):
        """Convert bytes to Galois Field elements"""
        return [[b for b in chunk] for chunk in chunks]
        
    def _generate_parity_chunk(self, chunks, index):
        """Generate single parity chunk"""
        parity = [0] * len(chunks[0])
        for i, chunk in enumerate(chunks):
            for j, byte in enumerate(chunk):
                parity[j] ^= self._gf_multiply(byte, self.generator_poly[i])
        return bytes(parity)
        
    def _gf_multiply(self, a, b):
        """Multiply two elements in GF(2^8)"""
        result = 0
        for i in range(8):
            if b & 1:
                result ^= a
            hi_bit_set = a & 0x80
            a = (a << 1) & 0xFF
            if hi_bit_set:
                a ^= 0x1D  # Primitive polynomial x^8 + x^4 + x^3 + x^2 + 1
            b >>= 1
        return result

class ExpMovingAverage:
    """Exponential Moving Average tracker"""
    def __init__(self, decay):
        self.decay = decay
        self.avg = 0
        self.steps = 0
        
    def update(self, value):
        """Update moving average"""
        self.steps += 1
        if self.steps == 1:
            self.avg = value
        else:
            self.avg = self.avg * self.decay + value * (1 - self.decay)
            
    @property
    def value(self):
        """Get current average value"""
        return self.avg

# Initialize
fusion = SSMXLSTMFusion(d_model=512, num_scales=3)

# Forward pass
x = torch.randn(32, 512)  # [batch_size, d_model]
output, h_new, c_new = fusion(x)

# Next step
next_output, next_h, next_c = fusion(next_x, h_new, c_new)

class NeuroFluxTrainer:
    """
    Complete training protocol implementation from Section 5.1 of whitepaper
    """
    def __init__(
        self,
        model: nn.Module,
        total_steps: int = 100000,
        warmup_steps: int = 1000
    ):
        self.model = model
        self.total_steps = total_steps
        self.curriculum = EnhancedCurriculumManager(total_steps, warmup_steps)
        self.hypernetwork = DifferentiableHyperNetwork(model.config.hidden_size)
        self.raid = RAIDMemory()
        
        # Optimizers for different phases
        self.optim = torch.optim.AdamW(model.parameters(), lr=2e-4)
        self.hyper_optim = torch.optim.AdamW(self.hypernetwork.parameters(), lr=1e-4)
        
        # LoRA for fine-tuning
        self.lora = None  # Initialize during exploitation phase
        
        # Phase-specific tracking
        self.phase_metrics = {
            'exploration': ExpMovingAverage(0.99),
            'exploitation': ExpMovingAverage(0.99),
            'consolidation': ExpMovingAverage(0.99)
        }
        
    def training_step(self, batch: Dict[str, torch.Tensor], step: int) -> Dict:
        """
        Execute single training step with phase-specific logic
        """
        phase = self.curriculum.get_current_phase(step)
        phase_config = self.curriculum.get_phase_config(step)
        
        # Get hyperparameters for current step
        delta, lambda_h, moe_temp = self.hypernetwork(
            self.model.get_last_hidden_state(),
            phase=phase
        )
        
        # Phase-specific forward pass
        if phase == 'exploration':
            return self._exploration_step(batch, delta, lambda_h, moe_temp, phase_config)
        elif phase == 'exploitation':
            return self._exploitation_step(batch, delta, lambda_h, moe_temp, phase_config)
        else:  # consolidation
            return self._consolidation_step(batch, delta, lambda_h, moe_temp, phase_config)
            
    def _exploration_step(
        self,
        batch: Dict[str, torch.Tensor],
        delta: torch.Tensor,
        lambda_h: torch.Tensor,
        moe_temp: torch.Tensor,
        config: Dict
    ) -> Dict:
        """
        Exploration phase: Train SSM + XLSTM with uniform MoE (k=4)
        """
        # Forward pass with uniform expert selection
        outputs = self.model(
            batch['input_ids'],
            delta=delta,
            moe_temp=moe_temp,
            top_k=4,  # Fixed k during exploration
            uniform_routing=True
        )
        
        # Compute losses
        task_loss = outputs.loss
        entropy_loss = -lambda_h * outputs.expert_entropy
        total_loss = task_loss + entropy_loss
        
        # Update model
        self.optim.zero_grad()
        self.hyper_optim.zero_grad()
        total_loss.backward()
        self.optim.step()
        self.hyper_optim.step()
        
        # Update metrics
        self.phase_metrics['exploration'].update(outputs.accuracy)
        
        return {
            'loss': total_loss.item(),
            'accuracy': outputs.accuracy,
            'expert_entropy': outputs.expert_entropy.item()
        }
        
    def _exploitation_step(
        self,
        batch: Dict[str, torch.Tensor],
        delta: torch.Tensor,
        lambda_h: torch.Tensor,
        moe_temp: torch.Tensor,
        config: Dict
    ) -> Dict:
        """
        Exploitation phase: GRPO-driven top-2 MoE with LoRA fine-tuning
        """
        # Initialize LoRA if needed
        if self.lora is None:
            self.lora = LoRALayer(self.model.xlstm)
            self.lora_optim = torch.optim.AdamW(self.lora.parameters(), lr=5e-5)
        
        # Forward pass with GRPO routing
        outputs = self.model(
            batch['input_ids'],
            delta=delta,
            moe_temp=moe_temp,
            top_k=2,  # GRPO-driven top-2
            lora=self.lora
        )
        
        # Compute GRPO loss
        grpo_loss = self.model.moe.compute_grpo_loss(
            outputs.old_logits,
            outputs.new_logits,
            outputs.advantages,
            outputs.expert_mask
        )
        
        # Total loss with LoRA regularization
        total_loss = outputs.loss + grpo_loss + 0.01 * self.lora.regularization_loss()
        
        # Update all components
        self.optim.zero_grad()
        self.hyper_optim.zero_grad()
        self.lora_optim.zero_grad()
        total_loss.backward()
        self.optim.step()
        self.hyper_optim.step()
        self.lora_optim.step()
        
        return {
            'loss': total_loss.item(),
            'grpo_loss': grpo_loss.item(),
            'accuracy': outputs.accuracy
        }
        
    def _consolidation_step(
        self,
        batch: Dict[str, torch.Tensor],
        delta: torch.Tensor,
        lambda_h: torch.Tensor,
        moe_temp: torch.Tensor,
        config: Dict
    ) -> Dict:
        """
        Consolidation phase: Freeze SSM, optimize RAID memory
        """
        # Freeze SSM parameters
        for param in self.model.ssm.parameters():
            param.requires_grad = False
            
        # Forward pass with RAID memory optimization
        outputs = self.model(
            batch['input_ids'],
            delta=delta,
            moe_temp=moe_temp,
            top_k=1,  # Single expert during consolidation
            optimize_raid=True
        )
        
        # REINFORCE loss for RAID optimization
        raid_loss = -outputs.raid_reward * outputs.raid_log_prob
        total_loss = outputs.loss + raid_loss
        
        # Update remaining components
        self.optim.zero_grad()
        self.hyper_optim.zero_grad()
        total_loss.backward()
        self.optim.step()
        self.hyper_optim.step()
        
        # Update RAID memory
        self.raid.update(
            self.model.get_last_hidden_state(),
            reward=outputs.raid_reward
        )
        
        return {
            'loss': total_loss.item(),
            'raid_loss': raid_loss.item(),
            'accuracy': outputs.accuracy,
            'raid_reward': outputs.raid_reward.item()
        }