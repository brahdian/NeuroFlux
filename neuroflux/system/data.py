import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
import random
from collections import defaultdict

@dataclass
class BatchConfig:
    """Configuration for dynamic batching"""
    max_tokens: int = 12288  # Maximum tokens per batch
    min_sequences: int = 4   # Minimum sequences per batch
    max_sequences: int = 64  # Maximum sequences per batch
    length_multiplier: int = 8  # Sequence length must be multiple of this
    target_tokens: int = 8192  # Target tokens per batch

class DynamicBatcher:
    """
    Implements dynamic batching strategy from Section 4.1
    Optimizes memory usage while maintaining throughput
    """
    def __init__(self, config: BatchConfig):
        self.config = config
        self.length_buckets = defaultdict(list)
        
    def add_sequence(self, sequence: torch.Tensor, idx: int):
        """Add sequence to appropriate length bucket"""
        length = sequence.size(0)
        # Round up to nearest multiple of length_multiplier
        padded_length = (
            (length + self.config.length_multiplier - 1) // 
            self.config.length_multiplier * 
            self.config.length_multiplier
        )
        self.length_buckets[padded_length].append((sequence, idx))
        
    def get_batch(self) -> Optional[Tuple[torch.Tensor, List[int]]]:
        """Get optimally sized batch"""
        # Try to find bucket that can form good batch
        for length, sequences in sorted(self.length_buckets.items()):
            if not sequences:
                continue
                
            # Calculate optimal batch size for this length
            max_sequences = min(
                self.config.max_sequences,
                self.config.max_tokens // length,
                len(sequences)
            )
            
            if max_sequences >= self.config.min_sequences:
                # Get batch
                batch_size = min(
                    max_sequences,
                    self.config.target_tokens // length
                )
                batch_sequences, batch_indices = zip(
                    *sequences[:batch_size]
                )
                
                # Remove used sequences
                self.length_buckets[length] = sequences[batch_size:]
                
                # Stack sequences
                return torch.stack(batch_sequences), list(batch_indices)
                
        return None

class NeuroFluxDataset(Dataset):
    """
    Dataset implementation with adaptive sequence handling
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        max_length: int = 2048,
        min_length: int = 32,
        token_mixing: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.token_mixing = token_mixing
        
        # Load and preprocess data
        self.sequences = self._load_data(data_path)
        self.sequence_lengths = [len(seq) for seq in self.sequences]
        
        # Initialize token mixing if enabled
        if token_mixing:
            self.mixing_buffer = []
            self.mixing_threshold = 1024  # Minimum tokens for mixing
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Apply token mixing if enabled
        if self.token_mixing and len(sequence) >= self.mixing_threshold:
            sequence = self._apply_token_mixing(sequence)
        
        # Truncate if necessary
        if len(sequence) > self.max_length:
            start_idx = random.randint(0, len(sequence) - self.max_length)
            sequence = sequence[start_idx:start_idx + self.max_length]
        
        return {
            'input_ids': sequence,
            'attention_mask': torch.ones_like(sequence),
            'length': len(sequence)
        }
    
    def _load_data(self, data_path: str) -> List[torch.Tensor]:
        """Load and preprocess data"""
        # Implementation depends on data format
        # This is a placeholder for actual data loading
        pass
    
    def _apply_token_mixing(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply token mixing strategy from Section 4.1"""
        # Add to mixing buffer
        self.mixing_buffer.append(sequence)
        
        if len(self.mixing_buffer) >= 4:  # Mix when we have enough sequences
            # Randomly select mixing points
            mix_points = sorted(random.sample(
                range(len(sequence)),
                random.randint(1, 3)
            ))
            
            # Create mixed sequence
            mixed_sequence = []
            start_idx = 0
            
            for mix_point in mix_points:
                # Add segment from current sequence
                mixed_sequence.append(sequence[start_idx:mix_point])
                
                # Add segment from random sequence in buffer
                other_seq = random.choice(self.mixing_buffer)
                other_length = min(
                    random.randint(32, 256),
                    len(other_seq)
                )
                other_start = random.randint(0, len(other_seq) - other_length)
                mixed_sequence.append(
                    other_seq[other_start:other_start + other_length]
                )
                
                start_idx = mix_point
            
            # Add final segment
            mixed_sequence.append(sequence[start_idx:])
            
            # Concatenate segments
            sequence = torch.cat(mixed_sequence)
            
            # Clear buffer occasionally
            if random.random() < 0.1:
                self.mixing_buffer = []
                
        return sequence

class DataCollator:
    """
    Custom collation with dynamic padding and length binning
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        pad_to_multiple: int = 8
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple = pad_to_multiple
        
    def __call__(
        self,
        features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        # Get max length in batch
        max_length = max(f['length'] for f in features)
        
        # Round up to multiple of pad_to_multiple
        max_length = (
            (max_length + self.pad_to_multiple - 1) //
            self.pad_to_multiple *
            self.pad_to_multiple
        )
        
        # Pad sequences
        input_ids = []
        attention_mask = []
        
        for f in features:
            padding_length = max_length - f['length']
            
            input_ids.append(torch.cat([
                f['input_ids'],
                torch.full((padding_length,), self.tokenizer.pad_token_id)
            ]))
            
            attention_mask.append(torch.cat([
                f['attention_mask'],
                torch.zeros(padding_length)
            ]))
        
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask)
        }

def create_dataloader(
    dataset: NeuroFluxDataset,
    batch_config: BatchConfig,
    tokenizer: PreTrainedTokenizer,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create dataloader with dynamic batching"""
    return DataLoader(
        dataset,
        batch_sampler=DynamicBatchSampler(
            dataset.sequence_lengths,
            batch_config
        ),
        collate_fn=DataCollator(tokenizer),
        num_workers=num_workers,
        pin_memory=True
    )

class DynamicBatchSampler:
    """
    Batch sampler that groups similar-length sequences
    """
    def __init__(
        self,
        lengths: List[int],
        config: BatchConfig,
        shuffle: bool = True
    ):
        self.lengths = lengths
        self.config = config
        self.shuffle = shuffle
        
        # Create length buckets
        self.buckets = defaultdict(list)
        for idx, length in enumerate(lengths):
            bucket = length // 64  # Group sequences into ~64-token buckets
            self.buckets[bucket].append(idx)
    
    def __iter__(self):
        # Shuffle buckets if needed
        if self.shuffle:
            for bucket in self.buckets.values():
                random.shuffle(bucket)
        
        # Create batches from each bucket
        batcher = DynamicBatcher(self.config)
        
        for bucket_indices in self.buckets.values():
            for idx in bucket_indices:
                sequence_length = self.lengths[idx]
                batcher.add_sequence(
                    torch.empty(sequence_length),  # Placeholder for length
                    idx
                )
                
            # Get all possible batches from this bucket
            while True:
                batch = batcher.get_batch()
                if batch is None:
                    break
                yield batch[1]  # Return indices
    
    def __len__(self):
        # Estimate number of batches
        total_tokens = sum(self.lengths)
        return total_tokens // self.config.target_tokens 