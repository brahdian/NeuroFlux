import pytest
import torch
from neuroflux.raid import RAIDMemory, GF256

@pytest.fixture
def raid():
    return RAIDMemory()

def test_encoding_decoding(raid):
    """Test RAID encoding and decoding"""
    # Create test state
    state = {
        'layer1': torch.randn(32, 128),
        'layer2': torch.randn(128, 64)
    }
    
    # Encode state
    encoded = raid.encode_memory(state)
    
    # Decode state
    decoded = raid.decode_memory(encoded)
    
    # Check reconstruction
    for key in state:
        assert torch.allclose(state[key], decoded[key], rtol=1e-5)

def test_failure_recovery(raid):
    """Test recovery from simulated failures"""
    # Original state
    state = torch.randn(16, 64)
    
    # Encode and corrupt some data
    encoded = raid.encode_memory({'test': state})
    encoded['data'][0] = torch.randn_like(encoded['data'][0])
    
    # Recover
    recovered = raid.decode_memory(encoded)
    
    # Check recovery
    assert torch.allclose(state, recovered['test'], rtol=1e-5)

def test_compression(raid):
    """Test FP8 compression"""
    x = torch.randn(32, 128)
    compressed = raid.compress_state(x)
    decompressed = raid.decompress_state(compressed)
    
    # Check compression ratio
    assert compressed['data'].nelement() < x.nelement()
    
    # Check reconstruction quality
    assert torch.allclose(x, decompressed, rtol=1e-2) 