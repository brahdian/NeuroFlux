import pytest
import torch
import numpy as np
from neuroflux.core.model import SSMXLSTMFusion
from neuroflux.utils.config_registry import ConfigRegistry

class TestSSMXLSTMFusion:
    """
    Comprehensive tests for SSM-XLSTM Fusion implementation
    
    Test Categories:
    - State transitions and memory fusion
    - Multi-scale processing
    - RAID integration
    - Edge cases and error handling
    - Performance characteristics
    """
    
    @pytest.fixture
    def config(self):
        config = ConfigRegistry.get_config()
        config.D_MODEL = 256
        config.N_HEADS = 8
        config.N_EXPERTS = 4
        config.XLSTM_SCALES = 3
        return config
    
    @pytest.fixture
    def model(self, config):
        return SSMXLSTMFusion()
    
    def test_state_transitions(self, model):
        """Test state transitions and memory updates"""
        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, model.config.D_MODEL)
        
        # Test with zero initial state
        output1, state1 = model(x)
        assert output1.shape == (batch_size, seq_len, model.config.D_MODEL)
        
        # Test state evolution
        output2, state2 = model(x, state1)
        assert not torch.allclose(state1, state2)
        
        # Test long sequence stability
        long_seq = torch.randn(batch_size, 1000, model.config.D_MODEL)
        output_long, state_long = model(long_seq)
        assert not torch.isnan(output_long).any()
        assert not torch.isinf(output_long).any()
    
    def test_multi_scale_processing(self, model):
        """Test multi-scale XLSTM behavior"""
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, model.config.D_MODEL)
        
        # Get intermediate states
        with torch.no_grad():
            states = model.get_intermediate_states(x)
            
        # Check different timescales
        for scale in range(model.config.XLSTM_SCALES):
            assert not torch.allclose(states[scale], states[(scale + 1) % model.config.XLSTM_SCALES])
    
    def test_raid_integration(self, model):
        """Test RAID memory integration"""
        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, model.config.D_MODEL)
        
        # Normal operation
        output1, state1 = model(x)
        
        # Simulate memory corruption
        corrupted_state = state1.clone()
        corrupted_state[:, :10] = float('nan')
        
        # Should recover using RAID
        output2, state2 = model(x, corrupted_state)
        assert not torch.isnan(output2).any()
        
        # Test parity slot corruption
        model.raid_buffer[0] = torch.randn_like(model.raid_buffer[0])
        output3, state3 = model(x, state2)
        assert not torch.isnan(output3).any()
    
    def test_edge_cases(self, model):
        """Test various edge cases and error conditions"""
        batch_size, seq_len = 4, 16
        
        # Test empty batch
        x_empty = torch.randn(0, seq_len, model.config.D_MODEL)
        with pytest.raises(ValueError):
            _ = model(x_empty)
        
        # Test single timestep
        x_single = torch.randn(batch_size, 1, model.config.D_MODEL)
        output_single, _ = model(x_single)
        assert output_single.shape == (batch_size, 1, model.config.D_MODEL)
        
        # Test very long sequence
        x_long = torch.randn(batch_size, 10000, model.config.D_MODEL)
        output_long, _ = model(x_long)
        assert output_long.shape == (batch_size, 10000, model.config.D_MODEL)
        
        # Test with NaN inputs
        x_nan = torch.full((batch_size, seq_len, model.config.D_MODEL), float('nan'))
        with pytest.raises(RuntimeError):
            _ = model(x_nan)
        
        # Test with infinity inputs
        x_inf = torch.full((batch_size, seq_len, model.config.D_MODEL), float('inf'))
        with pytest.raises(RuntimeError):
            _ = model(x_inf)
            
        # Test with mismatched dimensions
        x_wrong_dim = torch.randn(batch_size, seq_len, model.config.D_MODEL + 1)
        with pytest.raises(ValueError):
            _ = model(x_wrong_dim)
    
    def test_gradient_flow(self, model):
        """Test gradient flow through all components"""
        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, model.config.D_MODEL)
        
        # Forward pass
        output, _ = model(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients for all components
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("seq_len", [8, 16, 32])
    def test_different_shapes(self, model, batch_size, seq_len):
        """Test model with different input shapes"""
        x = torch.randn(batch_size, seq_len, model.config.D_MODEL)
        output, state = model(x)
        
        assert output.shape == (batch_size, seq_len, model.config.D_MODEL)
        assert state.shape == (batch_size, model.config.D_MODEL)
    
    def test_memory_recovery(self, model):
        """Test memory recovery mechanisms"""
        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, model.config.D_MODEL)
        
        # Normal operation
        output1, state1 = model(x)
        
        # Simulate partial memory corruption
        corrupt_mask = torch.rand_like(state1) > 0.5
        corrupted_state = state1.clone()
        corrupted_state[corrupt_mask] = float('nan')
        
        # Should recover using RAID
        output2, state2 = model(x, corrupted_state)
        assert not torch.isnan(output2).any()
        assert not torch.isnan(state2).any()
        
        # Test recovery with multiple corrupted regions
        corrupted_state2 = state2.clone()
        corrupted_state2[:, :10] = float('nan')
        corrupted_state2[:, -10:] = float('inf')
        
        output3, state3 = model(x, corrupted_state2)
        assert not torch.isnan(output3).any()
        assert not torch.isinf(output3).any() 