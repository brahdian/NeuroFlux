from typing import Optional, Dict, Any
from dataclasses import dataclass
import threading
from .config import Config

class ConfigRegistry:
    """
    Thread-safe singleton config registry
    """
    _instance = None
    _lock = threading.Lock()
    _config: Optional[Config] = None
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    @classmethod
    def get_config(cls) -> Config:
        """Get current config"""
        if cls._config is None:
            cls._config = Config()
        return cls._config
    
    @classmethod
    def set_config(cls, config: Config):
        """Set new config"""
        cls._config = config
    
    @classmethod
    def update_config(cls, **kwargs):
        """Update existing config"""
        config = cls.get_config()
        config.update(**kwargs) 