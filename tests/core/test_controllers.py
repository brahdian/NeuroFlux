import pytest
import torch
import numpy as np
from neuroflux.core.controllers import GRPOMoE, Expert
from neuroflux.utils.config_registry import ConfigRegistry

class TestGRPOMoE:
    """
    Comprehensive tests for GRPOMoE implementation
    
    Test Categories:
    - Initialization and basic functionality
    - Expert routing mechanisms
    - Load balancing and capacity
    - Fault tolerance and recovery
    - Edge cases and error handling
    - Performance characteristics
    """
    
    @pytest.fixture
    def config(self):
        config = ConfigRegistry.get_config()
        config.D_MODEL = 256
        config.N_EXPERTS = 4
        config.D_FF = 1024
        return config
    
    @pytest.fixture
    def controller(self, config):
        return GRPOMoE()
    
    def test_expert_initialization(self, controller):
        """Test expert initialization and structure"""
        assert len(controller.experts) == controller.config.N_EXPERTS
        assert all(isinstance(e, Expert) for e in controller.experts)
        
        # Test expert parameter initialization
        for expert in controller.experts:
            assert expert.d_model == controller.config.D_MODEL
            assert expert.d_ff == controller.config.D_FF
            
    def test_routing_mechanism(self, controller):
        """Test routing mechanism under various conditions"""
        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, controller.config.D_MODEL)
        
        # Normal routing
        output = controller(x)
        assert output.shape == x.shape
        assert output.requires_grad
        
        # Test with different temperatures
        output_hot = controller(x, temperature=0.1)
        output_cold = controller(x, temperature=10.0)
        
        # Temperature should affect expert selection entropy
        hot_entropy = controller.compute_routing_entropy()
        cold_entropy = controller.compute_routing_entropy()
        assert hot_entropy > cold_entropy
        
    def test_load_balancing(self, controller):
        """Test load balancing across experts"""
        batch_size, seq_len = 32, 16
        x = torch.randn(batch_size, seq_len, controller.config.D_MODEL)
        
        with torch.no_grad():
            _ = controller(x)
            expert_counts = controller.get_expert_counts()
            total_tokens = batch_size * seq_len
            
            # Basic capacity checks
            assert all(count <= 0.5 * total_tokens for count in expert_counts)
            
            # Run multiple batches to test stability
            for _ in range(5):
                _ = controller(x)
                new_counts = controller.get_expert_counts()
                # Counts should not vary drastically
                assert all(abs(new - old) < 0.3 * total_tokens 
                         for new, old in zip(new_counts, expert_counts))
                expert_counts = new_counts
    
    def test_expert_dropout(self, controller):
        """Test expert dropout and recovery"""
        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, controller.config.D_MODEL)
        
        # Test eval mode
        controller.eval()
        with torch.no_grad():
            output1 = controller(x)
            
        # Test train mode
        controller.train()
        with torch.no_grad():
            output2 = controller(x)
            
        assert not torch.allclose(output1, output2)
        
        # Test with different dropout rates
        controller.expert_dropout = 0.5
        with torch.no_grad():
            output3 = controller(x)
        assert not torch.allclose(output2, output3)
        
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("seq_len", [8, 16, 32])
    def test_different_shapes(self, controller, batch_size, seq_len):
        """Test model with different input shapes"""
        x = torch.randn(batch_size, seq_len, controller.config.D_MODEL)
        output = controller(x)
        assert output.shape == x.shape
        
    def test_edge_cases(self, controller):
        """Test various edge cases and error conditions"""
        # Test empty batch
        x_empty = torch.randn(0, 16, controller.config.D_MODEL)
        with pytest.raises(ValueError):
            _ = controller(x_empty)
            
        # Test very large batch
        x_large = torch.randn(1000, 16, controller.config.D_MODEL)
        output = controller(x_large)
        assert output.shape == x_large.shape
        
        # Test with NaN inputs
        x_nan = torch.full((4, 16, controller.config.D_MODEL), float('nan'))
        with pytest.raises(RuntimeError):
            _ = controller(x_nan)
            
        # Test with infinity inputs
        x_inf = torch.full((4, 16, controller.config.D_MODEL), float('inf'))
        with pytest.raises(RuntimeError):
            _ = controller(x_inf)
            
    def test_expert_failure_recovery(self, controller):
        """Test recovery from expert failures"""
        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, controller.config.D_MODEL)
        
        # Simulate expert failure
        failed_expert_idx = 1
        controller.experts[failed_expert_idx] = None
        
        # Should still work with n-1 experts
        output = controller(x)
        assert output.shape == x.shape
        
        # Check load redistribution
        expert_counts = controller.get_expert_counts()
        assert expert_counts[failed_expert_idx] == 0
        assert sum(expert_counts) == batch_size * seq_len
        
    def test_gradient_flow(self, controller):
        """Test gradient flow and backpropagation"""
        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, controller.config.D_MODEL)
        
        # Forward pass
        output = controller(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for expert in controller.experts:
            for param in expert.parameters():
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()
                
    def test_performance_characteristics(self, controller):
        """Test performance characteristics and memory usage"""
        batch_size, seq_len = 32, 128
        x = torch.randn(batch_size, seq_len, controller.config.D_MODEL)
        
        # Memory usage test
        torch.cuda.reset_peak_memory_stats()
        output = controller(x)
        memory_used = torch.cuda.max_memory_allocated()
        
        # Should be roughly linear with input size
        x_large = torch.randn(batch_size * 2, seq_len, controller.config.D_MODEL)
        output_large = controller(x_large)
        memory_used_large = torch.cuda.max_memory_allocated()
        
        # Check memory scaling (should be roughly linear)
        ratio = memory_used_large / memory_used
        assert 1.5 < ratio < 2.5  # Allow some overhead

        