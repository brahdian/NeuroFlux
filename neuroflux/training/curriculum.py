# neuroflux/curriculum.py
import torch
import torch.nn as nn
class EnhancedCurriculumManager:
    """
    Enhanced curriculum learning with dynamic phase transitions and RAID integration
    """
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
        
        # Load balancing settings
        self.expert_usage = ExpertUsageTracker(num_experts=8)
        
    def get_phase_config(self, step, metrics=None):
        """Get phase-specific configuration with dynamic adjustments"""
        phase = self.get_current_phase(step)
        
        # Base configurations
        configs = {
            'exploration': {
                'moe': {
                    'top_k': 4,
                    'temperature': 1.0,
                    'load_balance_factor': 1.2
                },
                'raid': {
                    'checkpoint_freq': 300,  # 5 minutes
                    'compression': 'fp16',
                    'parity_slots': 3  # Extra redundancy
                }
            },
            'exploitation': {
                'moe': {
                    'top_k': 2,
                    'temperature': 0.8,
                    'load_balance_factor': 1.0
                },
                'raid': {
                    'checkpoint_freq': 900,  # 15 minutes
                    'compression': 'fp8',
                    'parity_slots': 2
                }
            },
            'consolidation': {
                'moe': {
                    'top_k': 1,
                    'temperature': 0.5,
                    'load_balance_factor': 0.8
                },
                'raid': {
                    'checkpoint_freq': 3600,  # 1 hour
                    'compression': 'fp8',
                    'parity_slots': 2
                }
            }
        }
        
        config = configs[phase]
        
        # Dynamic adjustments based on metrics
        if metrics:
            config = self._adjust_config(config, metrics, phase)
            
        return config
    
    def _adjust_config(self, config, metrics, phase):
        """Dynamically adjust configuration based on metrics"""
        # Adjust MoE settings based on expert usage
        expert_stats = self.expert_usage.get_stats()
        if expert_stats['load_imbalance'] > 0.2:
            config['moe']['load_balance_factor'] *= 1.2
            config['moe']['temperature'] *= 1.1
            
        # Adjust RAID settings based on error rates
        if metrics.get('error_rate', 0) > 0.1:
            config['raid']['checkpoint_freq'] = max(
                300,  # 5 minutes minimum
                config['raid']['checkpoint_freq'] * 0.8
            )
            
        return config
    
    def update_expert_usage(self, expert_indices, expert_weights):
        """Track and balance expert usage"""
        stats = self.expert_usage.update(expert_indices, expert_weights)
        
        # Get underutilized experts
        underutilized = self.expert_usage.get_underutilized_experts()
        
        # Return load balancing signals
        return {
            'stats': stats,
            'underutilized': underutilized,
            'load_balancing_needed': len(underutilized) > 0
        }