import pytest
import torch
import torch.nn as nn
from neuroflux.model import SSMXLSTMFusion
from neuroflux.raid import RAIDMemory

@pytest.fixture
def model_config():
    return {
        'd_model': 128,
        'n_layers': 2,
        'n_experts': 4,
        'xlstm_scales': 2
    }

@pytest.fixture
def model(model_config):
    return SSMXLSTMFusion(**model_config)

def test_model_initialization(model, model_config):
    """Test model initialization and architecture"""
    assert model.d_model == model_config['d_model']
    assert len(model.xlstm_cells) == model_config['xlstm_scales']
    assert isinstance(model.raid, RAIDMemory)

def test_forward_pass(model):
    """Test model forward pass"""
    batch_size = 4
    seq_len = 16
    x = torch.randn(batch_size, seq_len, model.d_model)
    
    # Test forward pass
    output, h_ssm, c_new = model(x)
    
    # Check output shapes
    assert output.shape == (batch_size, seq_len, model.d_model)
    assert h_ssm.shape == (batch_size, model.d_model)
    assert len(c_new) == model.num_scales
    
    # Check no NaN values
    assert not torch.isnan(output).any()

def test_state_recovery(model):
    """Test model state recovery after corruption"""
    # Normal forward pass
    x = torch.randn(2, 8, model.d_model)
    original_output, _, _ = model(x)
    
    # Corrupt some parameters
    with torch.no_grad():
        for param in model.parameters():
            if torch.rand(1).item() < 0.1:
                param.data.mul_(torch.randn_like(param.data))
    
    # Recover and check output
    model.raid.recover_from_failure()
    recovered_output, _, _ = model(x)
    
    # Check if recovered output is close to original
    assert torch.allclose(original_output, recovered_output, rtol=1e-2) 