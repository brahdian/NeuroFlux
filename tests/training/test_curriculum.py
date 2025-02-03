import pytest
from neuroflux.curriculum import EnhancedCurriculumManager

@pytest.fixture
def curriculum():
    return EnhancedCurriculumManager(
        total_steps=100000,
        warmup_steps=1000
    )

def test_phase_transitions(curriculum):
    """Test phase transitions and boundaries"""
    # Check initial phase
    assert curriculum.get_current_phase(0) == 'exploration'
    
    # Check phase transitions
    steps = [0, 30000, 60000, 90000]
    expected_phases = ['exploration', 'exploitation', 'consolidation', 'consolidation']
    
    for step, expected in zip(steps, expected_phases):
        assert curriculum.get_current_phase(step) == expected

def test_phase_configs(curriculum):
    """Test phase-specific configurations"""
    # Get configs for different phases
    explore_config = curriculum.get_phase_config(0)
    exploit_config = curriculum.get_phase_config(50000)
    consol_config = curriculum.get_phase_config(90000)
    
    # Check phase-specific settings
    assert explore_config['moe']['top_k'] == 4
    assert exploit_config['moe']['top_k'] == 2
    assert consol_config['moe']['top_k'] == 1 