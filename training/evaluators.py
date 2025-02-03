import torch
import torch.nn as nn
from typing import Dict, List, Optional
from datasets import load_dataset

from ..core.model import SSMXLSTMFusion
from ..utils.config import Config
from ..utils.monitoring import PerformanceMonitor 