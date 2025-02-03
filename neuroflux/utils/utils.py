# neuroflux/utils.py
import torch
import hashlib
import os
import time
import json
from typing import Dict, Any, Optional

class CheckpointManager:
    """
    Enhanced checkpoint manager with RAID integration and validation
    """
    def __init__(
        self,
        path: str = "checkpoints",
        max_to_keep: int = 5,
        save_freq_mins: int = 5
    ):
        self.path = path
        self.max_to_keep = max_to_keep
        self.save_freq_mins = save_freq_mins
        self.last_save_time = time.time()
        
        # Create checkpoint directory
        os.makedirs(path, exist_ok=True)
        
        # Load checkpoint metadata if exists
        self.metadata_path = os.path.join(path, "checkpoint_metadata.json")
        self.metadata = self._load_metadata()
        
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: Optional[Dict] = None,
        force: bool = False
    ) -> str:
        """
        Save checkpoint with RAID encoding and validation
        Returns checkpoint path if saved, None if skipped
        """
        current_time = time.time()
        
        # Check if save is needed
        if not force and (current_time - self.last_save_time) < (self.save_freq_mins * 60):
            return None
            
        # Prepare checkpoint data
        checkpoint = {
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'step': step,
            'metrics': metrics or {},
            'timestamp': current_time
        }
        
        # Generate checkpoint path with hash
        checkpoint_hash = self._generate_hash(checkpoint)
        checkpoint_name = f"checkpoint_{step}_{checkpoint_hash[:8]}.pt"
        checkpoint_path = os.path.join(self.path, checkpoint_name)
        
        # Save with RAID encoding
        try:
            encoded_data = model.raid.encode_memory(checkpoint)
            torch.save(encoded_data, checkpoint_path)
            
            # Update metadata
            self.metadata['checkpoints'].append({
                'path': checkpoint_path,
                'step': step,
                'hash': checkpoint_hash,
                'timestamp': current_time,
                'metrics': metrics
            })
            
            # Keep only max_to_keep recent checkpoints
            self._cleanup_old_checkpoints()
            
            # Save metadata
            self._save_metadata()
            
            self.last_save_time = current_time
            return checkpoint_path
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return None
    
    def load(
        self,
        path: Optional[str] = None,
        step: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint with validation and RAID recovery
        """
        # Determine which checkpoint to load
        if path is None and step is not None:
            path = self._find_checkpoint_by_step(step)
        elif path is None:
            path = self._get_latest_checkpoint()
            
        if not path:
            raise ValueError("No checkpoint found")
            
        try:
            # Load encoded data
            encoded_data = torch.load(path)
            
            # Verify checkpoint hash
            stored_hash = path.split('_')[-1].split('.')[0]
            if not self._verify_checkpoint(encoded_data, stored_hash):
                raise ValueError("Checkpoint validation failed")
                
            # Decode checkpoint
            checkpoint = model.raid.decode_memory(encoded_data)
            return checkpoint
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            # Attempt recovery from previous checkpoint
            return self._recover_from_previous(path)
    
    def _generate_hash(self, checkpoint: Dict) -> str:
        """Generate deterministic hash for checkpoint validation"""
        # Convert tensor states to bytes
        state_bytes = b''
        for k, v in checkpoint['model_state'].items():
            if isinstance(v, torch.Tensor):
                state_bytes += v.cpu().numpy().tobytes()
                
        return hashlib.sha256(state_bytes).hexdigest()
    
    def _verify_checkpoint(self, encoded_data: Dict, stored_hash: str) -> bool:
        """Verify checkpoint integrity"""
        computed_hash = self._generate_hash(encoded_data)
        return computed_hash.startswith(stored_hash)
    
    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {'checkpoints': []}
    
    def _save_metadata(self):
        """Save checkpoint metadata"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only max_to_keep recent ones"""
        checkpoints = self.metadata['checkpoints']
        if len(checkpoints) > self.max_to_keep:
            # Sort by step number
            checkpoints.sort(key=lambda x: x['step'])
            
            # Remove old checkpoints
            for checkpoint in checkpoints[:-self.max_to_keep]:
                try:
                    os.remove(checkpoint['path'])
                    checkpoints.remove(checkpoint)
                except OSError:
                    pass
    
    def _find_checkpoint_by_step(self, step: int) -> Optional[str]:
        """Find checkpoint path by step number"""
        for checkpoint in self.metadata['checkpoints']:
            if checkpoint['step'] == step:
                return checkpoint['path']
        return None
    
    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get most recent checkpoint path"""
        if not self.metadata['checkpoints']:
            return None
        return max(
            self.metadata['checkpoints'],
            key=lambda x: x['step']
        )['path']
    
    def _recover_from_previous(self, failed_path: str) -> Dict[str, Any]:
        """Attempt to recover from previous checkpoint"""
        checkpoints = self.metadata['checkpoints']
        checkpoints.sort(key=lambda x: x['step'])
        
        # Find previous checkpoint
        for checkpoint in reversed(checkpoints):
            if checkpoint['path'] != failed_path:
                return self.load(checkpoint['path'])
                
        raise ValueError("No valid checkpoint found for recovery")

class ValueHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return torch.clamp(self.net(x), -1, 3)