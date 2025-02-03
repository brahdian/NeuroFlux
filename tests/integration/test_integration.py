import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List

from neuroflux.model import SSMXLSTMFusion
from neuroflux.controllers import GRPOMoE
from neuroflux.raid import RAIDMemory
from neuroflux.hypernetwork import DifferentiableHyperNetwork
from neuroflux.data import NeuroFluxDataset, create_dataloader
from neuroflux.distributed import DistributedTrainer
from neuroflux.monitoring import PerformanceMonitor
from neuroflux.deployment import DeploymentManager

class TestSystemIntegration:
    @pytest.fixture(scope="class")
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def model(self):
        return SSMXLSTMFusion(
            d_model=128,
            n_layers=2,
            n_experts=4
        )
    
    @pytest.fixture
    def trainer(self, model):
        return DistributedTrainer(
            model=model,
            config=DistributedConfig(world_size=1),
            optimizer=torch.optim.Adam(model.parameters())
        )
    
    @pytest.fixture
    def monitor(self):
        return PerformanceMonitor(MonitoringConfig())
    
    def test_full_training_loop(
        self,
        model,
        trainer,
        monitor,
        temp_dir
    ):
        """Test complete training loop with all components"""
        # Setup data
        dataset = self._create_dummy_dataset()
        dataloader = create_dataloader(dataset, batch_size=4)
        
        # Training loop
        for step, batch in enumerate(dataloader):
            if step >= 10:  # Short test
                break
                
            # Forward pass
            metrics = trainer.train_step(batch, grad_acc_steps=1)
            
            # Update monitoring
            monitor.update_training_metrics(
                loss=metrics['loss'],
                throughput=len(batch['input_ids']),
                step=step
            )
            
            # Verify outputs
            assert 'loss' in metrics
            assert not torch.isnan(torch.tensor(metrics['loss']))
    
    def test_fault_recovery(self, model, trainer, temp_dir):
        """Test system recovery from simulated failures"""
        # Setup RAID
        raid = RAIDMemory()
        
        # Save initial state
        state = {
            'model': model.state_dict(),
            'step': 0
        }
        encoded_state = raid.encode_memory(state)
        
        # Simulate failure by corrupting data
        corrupted_state = encoded_state.copy()
        corrupted_state['data'][0] = torch.randn_like(
            corrupted_state['data'][0]
        )
        
        # Recover
        recovered_state = raid.decode_memory(corrupted_state)
        
        # Verify recovery
        for key in state['model']:
            assert torch.allclose(
                state['model'][key],
                recovered_state['model'][key],
                rtol=1e-5
            )
    
    def test_distributed_synchronization(self):
        """Test gradient synchronization in distributed setting"""
        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs for this test")
            
        # Setup distributed training
        world_size = 2
        model = SSMXLSTMFusion(d_model=128)
        
        def train_worker(rank):
            trainer = DistributedTrainer(
                model=model,
                config=DistributedConfig(
                    world_size=world_size,
                    rank=rank
                ),
                optimizer=torch.optim.Adam(model.parameters())
            )
            
            # Generate same data on all ranks
            torch.manual_seed(42)
            batch = {
                'input_ids': torch.randint(0, 1000, (4, 16)),
                'attention_mask': torch.ones(4, 16)
            }
            
            # Train step
            metrics = trainer.train_step(batch, grad_acc_steps=1)
            
            return metrics
            
        # Run on multiple processes
        import torch.multiprocessing as mp
        mp.spawn(train_worker, nprocs=world_size)
    
    def test_deployment_workflow(
        self,
        model,
        temp_dir
    ):
        """Test model deployment workflow"""
        # Setup deployment
        deployment = DeploymentManager(
            DeploymentConfig(
                model_name="test_model",
                version="v1",
                base_path=temp_dir
            )
        )
        
        # Deploy model
        version = deployment.deploy_model(
            model,
            artifacts={'test': True}
        )
        
        # Test inference
        inputs = torch.randint(0, 1000, (1, 16))
        result = deployment.handle_request(inputs, version=version)
        
        assert 'result' in result
        assert 'latency' in result
        
        # Cleanup
        deployment.cleanup()
    
    def test_stress_conditions(
        self,
        model,
        trainer,
        monitor
    ):
        """Test system under stress conditions"""
        # Generate large batch
        large_batch = {
            'input_ids': torch.randint(0, 1000, (32, 512)),
            'attention_mask': torch.ones(32, 512)
        }
        
        # Monitor resources during stress test
        initial_memory = torch.cuda.memory_allocated()
        
        for _ in range(5):
            metrics = trainer.train_step(large_batch, grad_acc_steps=1)
            monitor.update_training_metrics(
                loss=metrics['loss'],
                throughput=len(large_batch['input_ids']),
                step=_
            )
        
        final_memory = torch.cuda.memory_allocated()
        
        # Check for memory leaks
        assert (final_memory - initial_memory) < 1e6  # Less than 1MB leak
    
    def _create_dummy_dataset(self) -> NeuroFluxDataset:
        """Create dummy dataset for testing"""
        data = [
            torch.randint(0, 1000, (16,))
            for _ in range(100)
        ]
        return NeuroFluxDataset(data)

if __name__ == "__main__":
    pytest.main([__file__]) 