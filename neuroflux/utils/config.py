# neuroflux/config.py
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import torch

@dataclass
class Config:
    """
    Complete NeuroFlux configuration from whitepaper specifications
    """
    # Model Architecture
    D_MODEL: int = 512
    N_LAYERS: int = 6
    N_HEADS: int = 8
    N_EXPERTS: int = 8
    XLSTM_SCALES: int = 3
    DROPOUT: float = 0.1
    
    # SSM Configuration
    SSM_STATE_DIM: int = 64
    SSM_DT_MIN: float = 0.001
    SSM_DT_MAX: float = 0.1
    
    # Training
    BATCH_SIZE: int = 32
    GRAD_ACC_STEPS: int = 4
    TOTAL_STEPS: int = 100_000
    WARMUP_STEPS: int = 1_000
    BASE_LR: float = 2e-4
    MIN_LR: float = 1e-5
    WEIGHT_DECAY: float = 0.1
    MAX_GRAD_NORM: float = 1.0
    
    # MoE Configuration
    EXPERT_CAPACITY: float = 1.25
    LOAD_BALANCE_DECAY: float = 0.999
    EXPERT_DROPOUT: float = 0.1
    GATING_TEMPERATURE: Tuple[float, float] = (0.5, 1.5)
    
    # RAID System
    CHECKPOINT_FREQ: int = 300  # 5 minutes
    MAX_CHECKPOINTS: int = 5
    PARITY_SLOTS: int = 2
    COMPRESSION_THRESHOLD: int = 1000
    RECOVERY_TIMEOUT: int = 360  # 6 minutes
    
    # Hypernetwork
    TRUST_REGION: bool = True
    DELTA_BOUNDS: Tuple[float, float] = (0.1, 2.0)
    LAMBDA_BOUNDS: Tuple[float, float] = (0.01, 0.99)
    HYPERNET_HIDDEN_DIM: int = 256
    
    # Curriculum Phases
    PHASE_RATIOS: Dict[str, float] = {
        'exploration': 0.3,
        'exploitation': 0.5,
        'consolidation': 0.2
    }
    
    # LoRA Settings
    LORA_RANK: int = 8
    LORA_ALPHA: float = 16
    LORA_DROPOUT: float = 0.1
    
    # Evaluation
    EVAL_BATCH_SIZE: int = 16
    CODE_GEN_TEMPERATURE: float = 0.8
    CODE_GEN_TOP_P: float = 0.95
    CODE_GEN_MAX_LENGTH: int = 512
    CODE_GEN_NUM_SAMPLES: int = 200
    CODE_GEN_TIMEOUT: int = 5
    
    # System
    CHECKPOINT_DIR: str = "/content/drive/MyDrive/neuroflux/checkpoints"
    LOG_DIR: str = "/content/drive/MyDrive/neuroflux/logs"
    NUM_GPUS: int = torch.cuda.device_count()
    FP16: bool = True
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Phase-specific configurations
    PHASE_CONFIGS: Dict[str, Dict] = {
        'exploration': {
            'moe': {
                'top_k': 4,
                'temperature': 1.0,
                'load_balance_factor': 0.01
            },
            'raid': {
                'checkpoint_freq': 300,  # 5 minutes
                'compression_ratio': 0.5
            },
            'trust_region': {
                'radius': 0.5
            }
        },
        'exploitation': {
            'moe': {
                'top_k': 2,
                'temperature': 0.8,
                'load_balance_factor': 0.1
            },
            'raid': {
                'checkpoint_freq': 600,  # 10 minutes
                'compression_ratio': 0.7
            },
            'trust_region': {
                'radius': 0.3
            }
        },
        'consolidation': {
            'moe': {
                'top_k': 1,
                'temperature': 0.5,
                'load_balance_factor': 0.2
            },
            'raid': {
                'checkpoint_freq': 900,  # 15 minutes
                'compression_ratio': 0.9
            },
            'trust_region': {
                'radius': 0.1
            }
        }
    }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create config from dictionary"""
        return cls(**{
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        })
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def get_phase_config(self, phase: str) -> Dict:
        """Get configuration for specific phase"""
        if phase not in self.PHASE_CONFIGS:
            raise ValueError(f"Invalid phase: {phase}")
        return self.PHASE_CONFIGS[phase]
    
    def update(self, **kwargs):
        """Update config parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid config parameter: {key}")

# Default configuration
default_config = Config()