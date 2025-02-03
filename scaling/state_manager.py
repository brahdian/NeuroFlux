"""
NeuroFlux State Manager
======================

Comprehensive state management system for long-term training pauses.
Ensures safe state preservation and seamless resumption.
"""

import torch
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import shutil
import pickle
from datetime import datetime

@dataclass
class TrainingState:
    """Complete training state information"""
    
    # Model and scaling info
    current_scale: str
    current_step: int
    total_steps: int
    parameter_count: int
    
    # Training progress
    last_loss: float
    best_loss: float
    learning_rate: float
    gradient_stats: Dict[str, float]
    
    # System state
    device_info: Dict[str, Any]
    memory_usage: Dict[str, float]
    
    # Timestamp
    pause_time: str = None
    
    def __post_init__(self):
        if not self.pause_time:
            self.pause_time = datetime.now().isoformat()

class StateManager:
    """Manages long-term state preservation and recovery"""
    
    def __init__(self, 
                 base_dir: str = "neuroflux_states",
                 redundancy_level: int = 3):
        self.base_dir = Path(base_dir)
        self.redundancy_level = redundancy_level
        self.logger = self._setup_logger()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("StateManager")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler("state_manager.log")
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        return logger
    
    def save_training_state(self, 
                          model: 'SSMXLSTMFusion',
                          scaler: 'NeuroFluxScaler',
                          optimizer: torch.optim.Optimizer,
                          training_info: Dict[str, Any]) -> str:
        """
        Save complete training state for long-term pause
        
        Returns:
            str: State ID for future recovery
        """
        # Generate unique state ID
        state_id = self._generate_state_id()
        state_dir = self.base_dir / state_id
        state_dir.mkdir(parents=True)
        
        # 1. Save model state with redundancy
        self._save_model_redundant(model, state_dir)
        
        # 2. Save optimizer state
        self._save_optimizer_state(optimizer, state_dir)
        
        # 3. Save scaler state
        self._save_scaler_state(scaler, state_dir)
        
        # 4. Create training state metadata
        state = TrainingState(
            current_scale=scaler.SCALE_STEPS[scaler.current_step].name,
            current_step=training_info.get('current_step', 0),
            total_steps=training_info.get('total_steps', 0),
            parameter_count=sum(p.numel() for p in model.parameters()),
            last_loss=training_info.get('last_loss', 0.0),
            best_loss=training_info.get('best_loss', 0.0),
            learning_rate=training_info.get('learning_rate', 0.0),
            gradient_stats=training_info.get('gradient_stats', {}),
            device_info=self._get_device_info(),
            memory_usage=self._get_memory_stats()
        )
        
        # 5. Save metadata
        self._save_metadata(state, state_dir)
        
        self.logger.info(f"Saved training state: {state_id}")
        self._print_pause_instructions(state_id)
        
        return state_id
    
    def load_training_state(self, state_id: str) -> Dict[str, Any]:
        """Recover training state after pause"""
        state_dir = self.base_dir / state_id
        if not state_dir.exists():
            raise ValueError(f"State {state_id} not found")
            
        # 1. Verify state integrity
        self._verify_state_integrity(state_dir)
        
        # 2. Load model state
        model_state = self._load_model_state(state_dir)
        
        # 3. Load optimizer state
        optimizer_state = self._load_optimizer_state(state_dir)
        
        # 4. Load scaler state
        scaler_state = self._load_scaler_state(state_dir)
        
        # 5. Load metadata
        metadata = self._load_metadata(state_dir)
        
        self.logger.info(f"Loaded training state: {state_id}")
        return {
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'scaler_state': scaler_state,
            'metadata': metadata
        }
    
    def _save_model_redundant(self, model: torch.nn.Module, state_dir: Path):
        """Save model state with redundancy"""
        model_state = model.state_dict()
        for i in range(self.redundancy_level):
            torch.save(
                model_state,
                state_dir / f"model_state_{i}.pt"
            )
    
    def _print_pause_instructions(self, state_id: str):
        """Print instructions for resuming training"""
        print("\n" + "="*50)
        print("Training State Saved Successfully!")
        print("="*50)
        print(f"\nState ID: {state_id}")
        print("\nTo resume training after your break:")
        print(f"1. Keep the directory: {self.base_dir}")
        print("2. Use this code to resume:")
        print("\nfrom neuroflux.scaling.state_manager import StateManager")
        print("state_manager = StateManager()")
        print(f"state = state_manager.load_training_state('{state_id}')")
        print("\nHave a great break! 🌟")
        print("="*50 + "\n")
    
    def _generate_state_id(self) -> str:
        """Generate unique state ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"state_{timestamp}_{random_suffix}"
    
    def verify_state(self, state_id: str) -> bool:
        """Verify state integrity"""
        try:
            state_dir = self.base_dir / state_id
            self._verify_state_integrity(state_dir)
            return True
        except Exception as e:
            self.logger.error(f"State verification failed: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize state manager
    state_manager = StateManager()
    
    # Save state before pause
    state_id = state_manager.save_training_state(
        model=current_model,
        scaler=current_scaler,
        optimizer=current_optimizer,
        training_info={
            'current_step': 10000,
            'total_steps': 100000,
            'last_loss': 2.34,
            'best_loss': 2.12,
            'learning_rate': 1e-4,
            'gradient_stats': {'mean': 0.01, 'std': 0.001}
        }
    )
    
    # After break, resume with:
    # state = state_manager.load_training_state(state_id) 