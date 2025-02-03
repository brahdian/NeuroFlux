import torch
import psutil
import GPUtil
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from collections import deque
import wandb
import json
import os
from datetime import datetime
from pathlib import Path
from .config_registry import ConfigRegistry

@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring"""
    sampling_rate: float = 1.0  # Hz
    history_size: int = 3600    # 1 hour of history
    alert_cooldown: int = 300   # 5 minutes between alerts
    gpu_temp_threshold: float = 80.0  # Celsius
    gpu_memory_threshold: float = 0.95  # 95% usage
    cpu_threshold: float = 0.90  # 90% usage
    memory_threshold: float = 0.85  # 85% usage
    throughput_drop_threshold: float = 0.3  # 30% drop
    log_dir: str = "logs"
    
    # New thresholds for enhanced monitoring
    gradient_norm_threshold: float = 100.0
    loss_spike_threshold: float = 5.0
    expert_imbalance_threshold: float = 0.3
    raid_recovery_time_threshold: float = 30.0  # seconds
    network_latency_threshold: float = 1.0  # seconds
    disk_usage_threshold: float = 0.90  # 90% usage

class PerformanceMonitor:
    """
    Enhanced performance monitoring system with comprehensive metrics and alerting
    
    Features:
    - Real-time system resource monitoring (CPU, GPU, Memory)
    - Training metrics tracking (loss, gradients, throughput)
    - Expert utilization analysis
    - RAID system health monitoring
    - Network performance tracking
    - Automated alerting system with multiple channels
    - Detailed metric logging and visualization
    
    Usage:
        monitor = PerformanceMonitor()
        monitor.start()
        
        # During training
        monitor.log_metrics({
            'loss': loss.item(),
            'gradient_norm': grad_norm,
            'expert_usage': expert_counts
        })
        
        # Cleanup
        monitor.cleanup()
    """
    def __init__(self):
        self.config = ConfigRegistry.get_config()
        self._trainer = None
        self._setup_logging()
        
        # Enhanced metrics history
        self.metrics_history = {
            # System metrics
            'gpu_util': deque(maxlen=self.config.history_size),
            'gpu_temp': deque(maxlen=self.config.history_size),
            'gpu_memory': deque(maxlen=self.config.history_size),
            'cpu_util': deque(maxlen=self.config.history_size),
            'memory_util': deque(maxlen=self.config.history_size),
            'disk_usage': deque(maxlen=self.config.history_size),
            
            # Training metrics
            'loss': deque(maxlen=self.config.history_size),
            'gradient_norm': deque(maxlen=self.config.history_size),
            'learning_rate': deque(maxlen=self.config.history_size),
            'throughput': deque(maxlen=self.config.history_size),
            
            # Expert metrics
            'expert_utilization': deque(maxlen=self.config.history_size),
            'expert_load_balance': deque(maxlen=self.config.history_size),
            'routing_entropy': deque(maxlen=self.config.history_size),
            
            # RAID metrics
            'raid_health': deque(maxlen=self.config.history_size),
            'recovery_time': deque(maxlen=self.config.history_size),
            'parity_check_time': deque(maxlen=self.config.history_size),
            
            # Network metrics
            'network_latency': deque(maxlen=self.config.history_size),
            'bandwidth_usage': deque(maxlen=self.config.history_size),
            'sync_time': deque(maxlen=self.config.history_size)
        }
        
        # Alert tracking with multiple channels
        self.alert_channels = {
            'email': self._send_email_alert,
            'slack': self._send_slack_alert,
            'console': self._send_console_alert,
            'wandb': self._send_wandb_alert
        }
        self.last_alert_time = {metric: 0 for metric in self.metrics_history}
        
        # Initialize monitoring thread
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Performance baselines
        self.baselines = {}
        self._initialize_baselines()

    def _initialize_baselines(self):
        """Initialize performance baselines for anomaly detection"""
        # System baselines
        self.baselines.update({
            'gpu_util_mean': 0.0,
            'memory_util_mean': 0.0,
            'throughput_mean': 0.0,
            'loss_mean': 0.0,
            'loss_std': 1.0
        })
        
    def _check_anomalies(self, metrics: Dict[str, float], current_time: float):
        """
        Enhanced anomaly detection with multiple alert levels
        
        Alert Levels:
        - WARNING: Potential issues that need attention
        - ERROR: Serious problems requiring immediate action
        - CRITICAL: System stability at risk
        """
        # System resource anomalies
        if metrics.get('gpu_util', 0) > self.config.gpu_threshold:
            self._alert('gpu_util', 'CRITICAL', f"GPU utilization critical: {metrics['gpu_util']:.1f}%", current_time)
            
        if metrics.get('memory_util', 0) > self.config.memory_threshold:
            self._alert('memory', 'ERROR', f"Memory usage high: {metrics['memory_util']:.1f}%", current_time)
            
        # Training anomalies
        if metrics.get('gradient_norm', 0) > self.config.gradient_norm_threshold:
            self._alert('gradient', 'WARNING', f"Gradient norm spike: {metrics['gradient_norm']:.1f}", current_time)
            
        loss = metrics.get('loss', 0)
        if abs(loss - self.baselines['loss_mean']) > self.config.loss_spike_threshold * self.baselines['loss_std']:
            self._alert('loss', 'ERROR', f"Unusual loss value: {loss:.3f}", current_time)
            
        # Expert utilization anomalies
        if metrics.get('expert_imbalance', 0) > self.config.expert_imbalance_threshold:
            self._alert('expert_balance', 'WARNING', f"Expert load imbalance: {metrics['expert_imbalance']:.2f}", current_time)
            
        # RAID system anomalies
        if metrics.get('raid_recovery_time', 0) > self.config.raid_recovery_time_threshold:
            self._alert('raid', 'CRITICAL', f"Slow RAID recovery: {metrics['raid_recovery_time']:.1f}s", current_time)
            
        # Network anomalies
        if metrics.get('network_latency', 0) > self.config.network_latency_threshold:
            self._alert('network', 'ERROR', f"High network latency: {metrics['network_latency']:.3f}s", current_time)

    def _alert(self, metric: str, level: str, message: str, current_time: float):
        """
        Enhanced alerting system with multiple channels and severity levels
        """
        if current_time - self.last_alert_time[metric] > self.config.alert_cooldown:
            self.last_alert_time[metric] = current_time
            
            # Format alert message
            alert_msg = f"[{level}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}"
            
            # Send to all configured channels
            for channel, sender in self.alert_channels.items():
                try:
                    sender(level, alert_msg)
                except Exception as e:
                    self.logger.error(f"Failed to send alert to {channel}: {e}")

    def get_detailed_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive metrics summary with statistical analysis
        """
        summary = {}
        
        for metric, values in self.metrics_history.items():
            if len(values) > 0:
                values_array = np.array(list(values))
                summary[metric] = {
                    'current': values_array[-1],
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'min': np.min(values_array),
                    'max': np.max(values_array),
                    'p95': np.percentile(values_array, 95),
                    'trend': self._compute_trend(values_array)
                }
                
        return summary

    def _compute_trend(self, values: np.ndarray) -> float:
        """Compute trend direction and magnitude using linear regression"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope

    @property
    def trainer(self):
        """Lazy load trainer to avoid circular imports"""
        if self._trainer is None:
            from ..system.distributed import DistributedTrainer
            self._trainer = DistributedTrainer
        return self._trainer
    
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('NeuroFluxMonitor')
        self.logger.setLevel(logging.INFO)
        
        os.makedirs(self.config.log_dir, exist_ok=True)
        log_path = os.path.join(
            self.config.log_dir,
            f'monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Update history
                for key, value in metrics.items():
                    self.metrics_history[key].append(value)
                
                # Check for anomalies
                self._check_anomalies(metrics, time.time())
                
                # Log metrics
                self._log_metrics(metrics)
                
                # Sleep until next sample
                time.sleep(1.0 / self.config.sampling_rate)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Back off on error
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect system metrics"""
        metrics = {}
        
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                metrics['gpu_util'] = np.mean([gpu.load for gpu in gpus])
                metrics['gpu_temp'] = np.mean([gpu.temperature for gpu in gpus])
                metrics['gpu_memory'] = np.mean([gpu.memoryUtil for gpu in gpus])
        except Exception as e:
            self.logger.warning(f"Failed to collect GPU metrics: {e}")
        
        # CPU metrics
        metrics['cpu_util'] = psutil.cpu_percent() / 100.0
        metrics['memory_util'] = psutil.virtual_memory().percent / 100.0
        
        return metrics
    
    def update_training_metrics(
        self,
        loss: float,
        throughput: float,
        step: int
    ):
        """Update training-specific metrics"""
        self.metrics_history['loss'].append(loss)
        self.metrics_history['throughput'].append(throughput)
        
        # Update baselines periodically
        if step % 100 == 0:
            self._update_baselines()
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'loss': loss,
                'throughput': throughput,
                'step': step
            })
    
    def _update_baselines(self):
        """Update performance baselines"""
        for metric in self.metrics_history:
            if len(self.metrics_history[metric]) > 0:
                self.baselines[metric] = np.mean(list(self.metrics_history[metric]))
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log current metrics"""
        if self.use_wandb:
            wandb.log(metrics)
        
        # Save detailed metrics periodically
        if time.time() % 300 < 1.0:  # Every 5 minutes
            self._save_detailed_metrics()
    
    def _save_detailed_metrics(self):
        """Save detailed metrics to file"""
        detailed_metrics = {
            metric: list(values)
            for metric, values in self.metrics_history.items()
        }
        
        metrics_path = os.path.join(
            self.config.log_dir,
            f'metrics_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
        )
        
        with open(metrics_path, 'w') as f:
            json.dump(detailed_metrics, f)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        
        for metric, values in self.metrics_history.items():
            if len(values) > 0:
                values_list = list(values)
                summary[metric] = {
                    'mean': np.mean(values_list),
                    'std': np.std(values_list),
                    'min': np.min(values_list),
                    'max': np.max(values_list),
                    'last': values_list[-1]
                }
        
        return summary
    
    def get_throttling_recommendation(self) -> Optional[Dict[str, float]]:
        """Get throttling recommendations based on system state"""
        summary = self.get_summary()
        
        if 'gpu_util' in summary and summary['gpu_util']['mean'] > 0.95:
            return {
                'batch_size_factor': 0.8,
                'gradient_accumulation_steps': 2
            }
        
        if 'memory_util' in summary and summary['memory_util']['mean'] > 0.9:
            return {
                'batch_size_factor': 0.7,
                'activation_checkpointing': True
            }
        
        return None
    
    def cleanup(self):
        """Cleanup monitoring resources"""
        self.running = False
        self.monitor_thread.join()
        self.logger.info("Monitoring system shutdown complete") 