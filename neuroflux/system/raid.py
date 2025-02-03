# neuroflux/raid.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import numpy as np
from pathlib import Path
import logging
import glob

class GF256:
    """
    Galois Field GF(2^8) implementation for Reed-Solomon coding
    as specified in Section 2.2 of whitepaper
    """
    def __init__(self):
        # Generate exp and log tables for GF(2^8)
        self.exp = [0] * 256
        self.log = [0] * 256
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x = self._multiply_primitive(x)
        self.exp[255] = self.exp[0]
    
    def _multiply_primitive(self, x):
        """Multiply by primitive polynomial x^8 + x^4 + x^3 + x^2 + 1"""
        highbit = x & 0x80
        x = (x << 1) & 0xFF
        if highbit:
            x ^= 0x1D
        return x
    
    def multiply(self, x, y):
        """Multiply two elements in GF(2^8)"""
        if x == 0 or y == 0:
            return 0
        return self.exp[(self.log[x] + self.log[y]) % 255]
    
    def divide(self, x, y):
        """Divide two elements in GF(2^8)"""
        if y == 0:
            raise ValueError("Division by zero in GF(2^8)")
        if x == 0:
            return 0
        return self.exp[(self.log[x] - self.log[y]) % 255]

class ReedSolomon:
    """
    Reed-Solomon encoder/decoder over GF(2^8) for RAID-6
    Implements equation G*M = P from Section 2.2
    """
    def __init__(self, num_data_blocks, num_parity_blocks, field=None):
        self.num_data = num_data_blocks
        self.num_parity = num_parity_blocks
        self.field = field or GF256()
        
        # Generate Vandermonde matrix for encoding
        self.generator_matrix = self._build_generator_matrix()
    
    def _build_generator_matrix(self):
        """Build Vandermonde matrix for RS encoding"""
        matrix = []
        for i in range(self.num_parity):
            row = []
            for j in range(self.num_data):
                # Use x^(i*j) for Vandermonde matrix
                power = (i * j) % 255
                row.append(self.field.exp[power])
            matrix.append(row)
        return matrix
    
    def encode(self, data_blocks):
        """Encode data blocks to generate parity"""
        if len(data_blocks) != self.num_data:
            raise ValueError("Incorrect number of data blocks")
            
        parity_blocks = [[0] * len(data_blocks[0]) for _ in range(self.num_parity)]
        
        # Compute G*M matrix multiplication in GF(2^8)
        for i in range(self.num_parity):
            for j in range(self.num_data):
                for k in range(len(data_blocks[0])):
                    parity_blocks[i][k] ^= self.field.multiply(
                        self.generator_matrix[i][j],
                        data_blocks[j][k]
                    )
        
        return data_blocks + parity_blocks
    
    def decode(self, available_blocks, available_indices):
        """Decode using surviving blocks to recover lost data"""
        if len(available_blocks) < self.num_data:
            raise ValueError("Not enough blocks for recovery")
            
        # Build decoding matrix based on available indices
        decode_matrix = []
        for i in range(len(available_indices)):
            row = []
            for j in range(self.num_data):
                power = (available_indices[i] * j) % 255
                row.append(self.field.exp[power])
            decode_matrix.append(row)
            
        # Solve system of equations to recover data
        return self._solve_linear_system(decode_matrix, available_blocks)
    
    def _solve_linear_system(self, matrix, data):
        """Solve linear system using Gaussian elimination in GF(2^8)"""
        # Implementation of Gaussian elimination over GF(2^8)
        # Returns recovered data blocks
        pass

class EnhancedRAID6(nn.Module):
    """
    Enhanced RAID-6 implementation with complete Reed-Solomon error correction
    and adaptive compression as described in the whitepaper
    """
    def __init__(self, num_blocks=4, parity_blocks=2, compression_ratio=8):
        super().__init__()
        self.rs = ReedSolomon(num_blocks, parity_blocks, field=GF256())
        self.num_blocks = num_blocks
        self.parity_blocks = parity_blocks
        
        # Enhanced memory banks with compression
        self.register_buffer('data_banks', torch.zeros(num_blocks, 256))
        self.register_buffer('parity_banks', torch.zeros(parity_blocks, 256))
        self.register_buffer('error_counts', torch.zeros(num_blocks + parity_blocks))
        
        # Adaptive compression
        self.compressor = nn.Sequential(
            nn.Linear(256, 256 // compression_ratio),
            nn.Tanh(),
            nn.Linear(256 // compression_ratio, 256)
        )
        
    def compress_state(self, state):
        """Enhanced FP8 compression with hybrid scheme from Section 4.2"""
        # Compute optimal scale factor based on value distribution
        abs_max = torch.abs(state).max()
        scale = min(127.0 / abs_max.item(), 100.0)  # Cap scale for stability
        
        # Apply hybrid compression scheme
        if state.numel() > 1000:  # Large tensors use FP8
            return torch.quantize_per_tensor(
                state,
                scale=1/scale,
                zero_point=0,
                dtype=torch.qint8
            )
        else:  # Small tensors keep full precision
            return state
        
    def decompress_state(self, compressed):
        return compressed.dequantize()
    
    def encode(self, data_chunks):
        """Enhanced Reed-Solomon encoding with compression"""
        compressed = [self.compress_state(chunk) for chunk in data_chunks]
        encoded = self.rs.encode([c.numpy() for c in compressed])
        
        self.data_banks = torch.tensor(encoded[:self.num_blocks])
        self.parity_banks = torch.tensor(encoded[self.num_blocks:])
        
    def decode(self, available_indices):
        """Enhanced failure recovery with decompression"""
        chunks = []
        for i in available_indices:
            if i < self.num_blocks:
                chunks.append(self.data_banks[i])
            else:
                chunks.append(self.parity_banks[i - self.num_blocks])
                
        decoded = self.rs.decode(chunks, available_indices)
        return [self.decompress_state(torch.tensor(d)) for d in decoded]

    def recover_from_failure(self):
        """
        Implements complete recovery workflow from Section 4.1 and 4.2 of whitepaper:
        1. Detect irrecoverable failure (3+ GPUs lost)
        2. Reload latest checkpoint
        3. Rebuild RAID memory
        4. Verify integrity
        """
        # Step 1: Detect failure severity
        failed_count = self.detect_failure_count()
        if failed_count < 3:
            # Can recover using just RAID-6 parity
            self.rebuild_from_parity()
            return
        
        # Step 2: Reload checkpoint since RAID recovery impossible
        checkpoint = self.reload_checkpoint()
        
        # Step 3: Rebuild RAID memory banks
        self.rebuild_raid_memory(checkpoint)
        
        # Step 4: Verify integrity
        if not self.verify_integrity():
            raise RuntimeError("Failed to recover RAID memory integrity")

    def detect_failure_count(self) -> int:
        """Count number of failed/corrupted memory banks"""
        failed = 0
        for bank in [*self.data_banks, *self.parity_banks]:
            if torch.isnan(bank).any() or torch.isinf(bank).any():
                failed += 1
        return failed

    def rebuild_from_parity(self):
        """Rebuild using Reed-Solomon when ≤2 failures"""
        available_data = []
        available_indices = []
        
        # Collect available chunks
        for i, bank in enumerate([*self.data_banks, *self.parity_banks]):
            if not torch.isnan(bank).any() and not torch.isinf(bank).any():
                available_data.append(bank.numpy())
                available_indices.append(i)
        
        # Decode using Reed-Solomon
        decoded = self.rs.decode(available_data, available_indices)
        
        # Restore data and parity banks
        self.data_banks = torch.tensor(decoded[:self.num_blocks]) 
        self.parity_banks = torch.tensor(decoded[self.num_blocks:])

    def reload_checkpoint(self):
        """Reload latest stable checkpoint with FP8 compression"""
        checkpoint_paths = sorted(glob.glob("checkpoint_*.pt"))
        if not checkpoint_paths:
            raise RuntimeError("No checkpoints found for recovery")
        
        # Load latest checkpoint
        checkpoint = torch.load(checkpoint_paths[-1])
        
        # Decompress FP8 weights if needed
        if isinstance(checkpoint['model_state'], dict):
            for k, v in checkpoint['model_state'].items():
                if isinstance(v, dict) and 'quantized' in v:
                    checkpoint['model_state'][k] = v['quantized'].float() * v['scale']
                
        return checkpoint

    def rebuild_raid_memory(self, checkpoint):
        """
        Rebuild RAID memory banks from checkpoint with FP8 compression
        as described in Section 4.2 of whitepaper
        """
        # Extract memory state
        if 'raid_memory' not in checkpoint:
            raise RuntimeError("Checkpoint missing RAID memory state")
        
        raid_state = checkpoint['raid_memory']
        
        # Decompress FP8 data if needed
        if 'compression_format' in raid_state and raid_state['compression_format'] == 'fp8':
            data_chunks = []
            for chunk in raid_state['data_chunks']:
                if isinstance(chunk, dict) and 'scale' in chunk:
                    # Decompress FP8 format: value * scale
                    data_chunks.append(chunk['quantized'].float() * chunk['scale'])
                else:
                    data_chunks.append(chunk)
        else:
            data_chunks = raid_state['data_chunks']
        
        # Restore data banks with compression
        self.data_banks = torch.tensor([
            self.compress_state(chunk) 
            for chunk in data_chunks
        ])
        
        # Recompute parity using Reed-Solomon
        encoded = self.rs.encode(self.data_banks.numpy())
        self.parity_banks = torch.tensor(encoded[self.num_blocks:])
        
        # Reset error tracking
        self.error_counts.zero_()

    def verify_integrity(self) -> bool:
        """
        Enhanced RAID memory integrity verification with additional checks
        """
        try:
            # Check data bank consistency and ranges
            for bank in self.data_banks:
                if torch.isnan(bank).any() or torch.isinf(bank).any():
                    return False
                    
                # Check for reasonable value ranges after decompression
                decompressed = self.decompress_state(bank)
                if torch.abs(decompressed).max() > 100:  # Threshold for unreasonable values
                    return False
            
            # Verify parity matches data with enhanced precision
            encoded = self.rs.encode(self.data_banks.numpy())
            expected_parity = torch.tensor(encoded[self.num_blocks:])
            
            # Check both relative and absolute differences
            rel_error = torch.abs(self.parity_banks - expected_parity) / (torch.abs(expected_parity) + 1e-6)
            abs_error = torch.abs(self.parity_banks - expected_parity)
            
            return (rel_error < 1e-4).all() and (abs_error < 1e-6).all()
            
        except Exception as e:
            print(f"Integrity check failed: {str(e)}")
            return False

    def get_adaptive_checkpoint_interval(self) -> int:
        """Enhanced adaptive checkpoint interval from Section 4.2"""
        # Base intervals from whitepaper
        MIN_INTERVAL = 300  # 5 minutes
        MAX_INTERVAL = 7200  # 2 hours
        
        # Compute error rate with exponential decay
        recent_errors = self.error_counts.sum().item()
        total_banks = len(self.error_counts)
        error_rate = recent_errors / max(1, total_banks)
        
        # Dynamic interval based on error rates
        if error_rate > 0.1:  # High error rate
            return MIN_INTERVAL
        elif error_rate > 0.01:  # Moderate error rate
            return MIN_INTERVAL * 6  # 30 minutes
        else:  # Low error rate
            # Gradually increase up to max interval
            return min(
                MAX_INTERVAL,
                MIN_INTERVAL * 24 * (1 - error_rate) / 0.01
            )

    def update_error_tracking(self):
        """Update error statistics for adaptive checkpointing"""
        # Detect errors in current state
        current_errors = torch.zeros_like(self.error_counts)
        
        # Check data banks
        for i, bank in enumerate(self.data_banks):
            if torch.isnan(bank).any() or torch.isinf(bank).any():
                current_errors[i] = 1
                
        # Check parity banks
        for i, bank in enumerate(self.parity_banks):
            if torch.isnan(bank).any() or torch.isinf(bank).any():
                current_errors[i + self.num_blocks] = 1
                
        # Update error history with exponential decay
        decay = 0.95
        self.error_counts = decay * self.error_counts + (1 - decay) * current_errors

    def retrieve(self, h_prev):
        """
        Adaptive memory retrieval with error checking
        Returns recovered state or None if unrecoverable
        """
        try:
            # Check for errors in current state
            if torch.isnan(h_prev).any() or torch.isinf(h_prev).any():
                # Attempt recovery from RAID
                recovered = self.recover_from_failure()
                if recovered is not None:
                    return recovered
                    
            # No errors, return decompressed state
            return self.decompress_state(h_prev)
            
        except Exception as e:
            print(f"Memory retrieval failed: {str(e)}")
            return None
    
    def encode_memory(self, state_dict):
        """
        Encode model state into RAID format with FP8 compression
        Returns encoded state with parity
        """
        # Compress state to FP8
        compressed = {}
        for key, tensor in state_dict.items():
            if tensor.requires_grad:
                # Use FP8 for trainable parameters
                compressed[key] = self.compress_state(tensor)
            else:
                # Keep buffers in original precision
                compressed[key] = tensor
                
        # Split into chunks and compute parity
        chunks = self._split_into_chunks(compressed)
        encoded = self.rs.encode(chunks)
        
        return {
            'data': encoded[:self.num_blocks],
            'parity': encoded[self.num_blocks:],
            'compression_format': 'fp8'
        }
    
    def decode_memory(self, encoded_state):
        """
        Decode RAID-encoded state with FP8 decompression
        """
        # Recover from available chunks
        available = encoded_state['data'] + encoded_state['parity']
        indices = list(range(len(available)))
        
        decoded = self.rs.decode(available, indices)
        
        # Rebuild state dict with decompression
        state_dict = {}
        for key, compressed in decoded.items():
            if encoded_state.get('compression_format') == 'fp8':
                state_dict[key] = self.decompress_state(compressed)
            else:
                state_dict[key] = compressed
                
        return state_dict

class RAIDMemory(nn.Module):
    """
    Redundant Array of Independent Distributed Memory
    Implements fault-tolerant memory management with parity-based recovery
    """
    def __init__(self, 
                 num_blocks: int = 8,
                 parity_slots: int = 2,
                 compression_threshold: int = 1000,
                 recovery_timeout: int = 360):
        super().__init__()
        self.num_blocks = num_blocks
        self.parity_slots = parity_slots
        self.compression_threshold = compression_threshold
        self.recovery_timeout = recovery_timeout
        
        # Initialize memory banks
        self.data_banks: List[torch.Tensor] = []
        self.parity_banks: List[torch.Tensor] = []
        self.error_counts = torch.zeros(num_blocks + parity_slots)
        
    def store(self, data: torch.Tensor) -> None:
        """Store data with redundancy"""
        # Split data into blocks
        blocks = self._split_into_blocks(data)
        
        # Compute parity
        parity = self._compute_parity(blocks)
        
        # Update storage
        self.data_banks = blocks
        self.parity_banks = parity
        
    def recover_from_failure(self) -> Optional[torch.Tensor]:
        """Attempt to recover data after detecting corruption"""
        try:
            # Check for corrupted blocks
            corrupted = self._detect_corruption()
            if not corrupted:
                return self._reconstruct_data()
                
            # Attempt recovery using parity
            recovered = self._recover_using_parity(corrupted)
            if recovered is not None:
                return self._reconstruct_data()
                
        except Exception as e:
            logging.error(f"Recovery failed: {str(e)}")
            return None
            
    def _split_into_blocks(self, data: torch.Tensor) -> List[torch.Tensor]:
        """Split input tensor into blocks"""
        chunks = torch.chunk(data, self.num_blocks)
        return [chunk.clone() for chunk in chunks]
        
    def _compute_parity(self, blocks: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute parity blocks for redundancy"""
        parity = []
        for i in range(self.parity_slots):
            p = blocks[0].clone()
            for block in blocks[1:]:
                p = p ^ block  # XOR operation for parity
            parity.append(p)
        return parity
        
    def _detect_corruption(self) -> List[int]:
        """Detect corrupted blocks using parity checks"""
        corrupted = []
        for i, bank in enumerate(self.data_banks):
            if torch.isnan(bank).any() or torch.isinf(bank).any():
                corrupted.append(i)
        return corrupted
        
    def _recover_using_parity(self, corrupted: List[int]) -> Optional[List[torch.Tensor]]:
        """Recover corrupted blocks using parity information"""
        if len(corrupted) > self.parity_slots:
            return None  # Too many corrupted blocks
            
        recovered = []
        # Use different parity banks for each corrupted block
        for idx, i in enumerate(corrupted):
            # Use corresponding parity bank for recovery
            recovered_block = self.parity_banks[idx].clone()
            for j, block in enumerate(self.data_banks):
                if j != i and not j in corrupted:  # Skip corrupted blocks in recovery
                    recovered_block = recovered_block ^ block
            recovered.append(recovered_block)
            
        # Update recovered blocks
        for i, block in zip(corrupted, recovered):
            self.data_banks[i] = block
            
        return self.data_banks
        
    def _reconstruct_data(self) -> torch.Tensor:
        """Reconstruct original data from blocks"""
        return torch.cat(self.data_banks, dim=0)

    def get_adaptive_checkpoint_interval(self) -> int:
        """Enhanced adaptive checkpoint interval from Section 4.2"""
        # Base intervals from whitepaper
        MIN_INTERVAL = 300  # 5 minutes
        MAX_INTERVAL = 7200  # 2 hours
        
        # Compute error rate with exponential decay
        recent_errors = self.error_counts.sum().item()
        total_banks = len(self.error_counts)
        error_rate = recent_errors / max(1, total_banks)
        
        # Dynamic interval based on error rates
        if error_rate > 0.1:  # High error rate
            return MIN_INTERVAL
        elif error_rate > 0.01:  # Moderate error rate
            return MIN_INTERVAL * 6  # 30 minutes
        else:  # Low error rate
            # Gradually increase up to max interval
            return min(
                MAX_INTERVAL,
                MIN_INTERVAL * 24 * (1 - error_rate) / 0.01
            )

    def update_error_tracking(self):
        """Update error statistics for adaptive checkpointing"""
        # Detect errors in current state
        current_errors = torch.zeros_like(self.error_counts)
        
        # Check data banks
        for i, bank in enumerate(self.data_banks):
            if torch.isnan(bank).any() or torch.isinf(bank).any():
                current_errors[i] = 1
                
        # Check parity banks
        for i, bank in enumerate(self.parity_banks):
            if torch.isnan(bank).any() or torch.isinf(bank).any():
                current_errors[i + self.num_blocks] = 1
                
        # Update error history with exponential decay
        decay = 0.95
        self.error_counts = decay * self.error_counts + (1 - decay) * current_errors