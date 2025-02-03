import pytest
import torch
from neuroflux.hypernetwork import DifferentiableHyperNetwork

@pytest.fixture
def hypernetwork():
    return DifferentiableHyperNetwork(d_model=128)

def test_parameter_bounds(hypernetwork):
    """Test parameter predictions stay within bounds"""
    x = torch.randn(4, hypernetwork.d_model)
    delta, lambda_h, moe_temp = hypernetwork(x)
    
    # Check delta bounds
    assert (delta >= hypernetwork.delta_net.bounds[0]).all()
    assert (delta <= hypernetwork.delta_net.bounds[1]).all()
    
    # Check lambda bounds
    assert (lambda_h >= hypernetwork.lambda_net.bounds[0]).all()
    assert (lambda_h <= hypernetwork.lambda_net.bounds[1]).all()

def test_trust_region(hypernetwork):
    """Test trust region constraints"""
    x = torch.randn(4, hypernetwork.d_model)
    
    # Get predictions for different phases
    params_explore = hypernetwork(x, phase='exploration')
    params_exploit = hypernetwork(x, phase='exploitation')
    params_consol = hypernetwork(x, phase='consolidation')
    
    # Check phase-specific constraints
    for phase_params in [params_explore, params_exploit, params_consol]:
        delta, lambda_h, moe_temp = phase_params
        assert not torch.isnan(delta).any()
        assert not torch.isnan(lambda_h).any()
        assert not torch.isnan(moe_temp).any() 