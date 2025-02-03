import torch
import os
import json
import time
import logging
import subprocess
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import requests
import docker
from pathlib import Path
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor

from ..utils.config_registry import ConfigRegistry

@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    model_name: str
    version: str
    base_path: str = "deployments"
    max_batch_size: int = 32
    timeout: float = 30.0  # seconds
    max_concurrent_requests: int = 100
    enable_versioning: bool = True
    enable_monitoring: bool = True
    enable_ab_testing: bool = False
    ab_test_traffic_split: float = 0.1  # 10% to new version
    health_check_interval: float = 60.0  # seconds
    auto_scale: bool = True
    min_replicas: int = 1
    max_replicas: int = 4
    target_gpu_utilization: float = 0.8

class DeploymentManager:
    """
    Deployment manager with improved dependency handling
    """
    def __init__(self, **kwargs):
        self.config = ConfigRegistry.get_config()
        # Override config with kwargs
        for k, v in kwargs.items():
            setattr(self.config, k, v)
            
        self._monitor = None
        self._setup_deployment()
    
    @property
    def monitor(self):
        """Lazy load monitor to avoid circular imports"""
        if self._monitor is None:
            from ..utils.monitoring import PerformanceMonitor
            self._monitor = PerformanceMonitor()
        return self._monitor
    
    def _setup_deployment(self):
        self.logger = self._setup_logger()
        
        # Initialize docker client
        self.docker_client = docker.from_env()
        
        # Version control
        self.versions = self._load_versions()
        
        # A/B testing state
        self.ab_test_state = {
            'active': False,
            'control_version': None,
            'test_version': None,
            'metrics': {}
        }
        
        # Active deployments
        self.active_deployments = {}
        
        # Health monitoring
        self.health_thread = threading.Thread(target=self._health_monitor)
        self.health_thread.daemon = True
        self.health_thread.start()
        
        # Request handling
        self.request_executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_requests
        )
    
    def deploy_model(
        self,
        model: torch.nn.Module,
        artifacts: Dict[str, Any],
        version: Optional[str] = None
    ) -> str:
        """Deploy new model version"""
        version = version or f"v{len(self.versions) + 1}"
        deploy_path = Path(self.config.base_path) / version
        
        try:
            # Save model and artifacts
            self._save_deployment(model, artifacts, deploy_path)
            
            # Update version control
            self.versions[version] = {
                'timestamp': datetime.now().isoformat(),
                'artifacts': artifacts,
                'status': 'deployed',
                'metrics': {}
            }
            self._save_versions()
            
            # Build and start container
            container = self._start_deployment(version)
            self.active_deployments[version] = container
            
            # Start health monitoring
            self._monitor_deployment(version)
            
            self.logger.info(f"Successfully deployed version {version}")
            return version
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            if version in self.versions:
                self.versions[version]['status'] = 'failed'
                self._save_versions()
            raise
    
    def start_ab_test(
        self,
        control_version: str,
        test_version: str,
        duration_days: float = 7.0
    ):
        """Start A/B test between two versions"""
        if not self.config.enable_ab_testing:
            raise ValueError("A/B testing is not enabled")
            
        self.ab_test_state.update({
            'active': True,
            'control_version': control_version,
            'test_version': test_version,
            'start_time': datetime.now(),
            'duration_days': duration_days,
            'metrics': {
                'control': {'requests': 0, 'latency': [], 'errors': 0},
                'test': {'requests': 0, 'latency': [], 'errors': 0}
            }
        })
        
        self.logger.info(
            f"Started A/B test: {control_version} (control) vs {test_version} (test)"
        )
    
    def handle_request(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        version: Optional[str] = None
    ) -> Dict:
        """Handle inference request with optional version routing"""
        # A/B test routing
        if (
            self.config.enable_ab_testing and
            self.ab_test_state['active'] and
            version is None
        ):
            if np.random.random() < self.config.ab_test_traffic_split:
                version = self.ab_test_state['test_version']
                test_type = 'test'
            else:
                version = self.ab_test_state['control_version']
                test_type = 'control'
        
        version = version or max(self.versions.keys())  # Latest version
        
        try:
            start_time = time.time()
            
            # Submit request to deployment
            future = self.request_executor.submit(
                self._forward_request,
                version,
                inputs
            )
            
            result = future.result(timeout=self.config.timeout)
            latency = time.time() - start_time
            
            # Update A/B test metrics
            if self.ab_test_state['active']:
                metrics = self.ab_test_state['metrics'][test_type]
                metrics['requests'] += 1
                metrics['latency'].append(latency)
            
            return {
                'result': result,
                'version': version,
                'latency': latency
            }
            
        except Exception as e:
            self.logger.error(f"Request failed for version {version}: {e}")
            if self.ab_test_state['active']:
                self.ab_test_state['metrics'][test_type]['errors'] += 1
            raise
    
    def _save_deployment(
        self,
        model: torch.nn.Module,
        artifacts: Dict[str, Any],
        path: Path
    ):
        """Save model and deployment artifacts"""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), path / "model.pt")
        
        # Save config
        with open(path / "config.yaml", 'w') as f:
            yaml.dump(artifacts, f)
        
        # Save Dockerfile
        self._generate_dockerfile(path)
    
    def _generate_dockerfile(self, path: Path):
        """Generate Dockerfile for deployment"""
        dockerfile = f"""
FROM pytorch/pytorch:latest

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["python", "serve.py"]
"""
        with open(path / "Dockerfile", 'w') as f:
            f.write(dockerfile)
    
    def _start_deployment(self, version: str) -> docker.models.containers.Container:
        """Start containerized deployment"""
        deploy_path = Path(self.config.base_path) / version
        
        # Build image
        image, _ = self.docker_client.images.build(
            path=str(deploy_path),
            tag=f"{self.config.model_name}:{version}"
        )
        
        # Start container
        container = self.docker_client.containers.run(
            image.id,
            detach=True,
            ports={'8000/tcp': None},
            gpus='all',
            environment={
                'MODEL_VERSION': version,
                'MAX_BATCH_SIZE': str(self.config.max_batch_size)
            }
        )
        
        return container
    
    def _monitor_deployment(self, version: str):
        """Monitor deployment health and performance"""
        def check_health():
            try:
                container = self.active_deployments[version]
                stats = container.stats(stream=False)
                
                # Check GPU utilization
                if self.config.auto_scale:
                    self._handle_auto_scaling(version, stats)
                
                return True
            except Exception as e:
                self.logger.error(f"Health check failed for {version}: {e}")
                return False
        
        return check_health()
    
    def _handle_auto_scaling(self, version: str, stats: Dict):
        """Handle auto-scaling based on utilization"""
        gpu_util = stats.get('gpu_stats', {}).get('utilization', 0)
        current_replicas = len(self._get_version_containers(version))
        
        if gpu_util > self.config.target_gpu_utilization and current_replicas < self.config.max_replicas:
            self._scale_deployment(version, current_replicas + 1)
        elif gpu_util < self.config.target_gpu_utilization * 0.7 and current_replicas > self.config.min_replicas:
            self._scale_deployment(version, current_replicas - 1)
    
    def _scale_deployment(self, version: str, target_replicas: int):
        """Scale deployment to target number of replicas"""
        current_replicas = len(self._get_version_containers(version))
        
        if target_replicas > current_replicas:
            # Scale up
            for _ in range(target_replicas - current_replicas):
                container = self._start_deployment(version)
                self.active_deployments[f"{version}_replica_{_}"] = container
        else:
            # Scale down
            containers = self._get_version_containers(version)
            for container in containers[target_replicas:]:
                container.stop()
                container.remove()
                del self.active_deployments[container.name]
    
    def _get_version_containers(self, version: str) -> List[docker.models.containers.Container]:
        """Get all containers for a version"""
        return [
            container for name, container in self.active_deployments.items()
            if version in name
        ]
    
    def get_ab_test_results(self) -> Dict:
        """Get current A/B test results"""
        if not self.ab_test_state['active']:
            return {}
            
        metrics = self.ab_test_state['metrics']
        
        def compute_stats(data):
            return {
                'requests': data['requests'],
                'errors': data['errors'],
                'error_rate': data['errors'] / max(1, data['requests']),
                'avg_latency': np.mean(data['latency']) if data['latency'] else 0,
                'p95_latency': np.percentile(data['latency'], 95) if data['latency'] else 0
            }
        
        return {
            'control': compute_stats(metrics['control']),
            'test': compute_stats(metrics['test']),
            'duration': (datetime.now() - self.ab_test_state['start_time']).total_seconds() / 86400
        }
    
    def cleanup(self):
        """Cleanup deployment resources"""
        for container in self.active_deployments.values():
            try:
                container.stop()
                container.remove()
            except Exception as e:
                self.logger.error(f"Error cleaning up container: {e}")
        
        self.request_executor.shutdown()
        self.logger.info("Deployment cleanup complete")
    
    def _load_versions(self) -> Dict:
        """Load version history"""
        version_file = Path(self.config.base_path) / "versions.json"
        if version_file.exists():
            with open(version_file) as f:
                return json.load(f)
        return {}
    
    def _save_versions(self):
        """Save version history"""
        version_file = Path(self.config.base_path) / "versions.json"
        with open(version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('NeuroFluxDeployment')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        
        return logger 