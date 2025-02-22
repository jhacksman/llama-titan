"""
Memory Management System for the Titans architecture.
Handles VRAM budget tracking and memory sharding across GPUs.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

class MemoryManager:
    """Manages memory distribution and VRAM usage across components."""
    
    def __init__(self, config):
        self.vram_budget = 64 * 1024 * 1024 * 1024  # 64GB in bytes
        self.current_usage = 0
        self.component_budgets = {
            'core': 30 * 1024 * 1024 * 1024,      # 30GB
            'long_term': 15 * 1024 * 1024 * 1024,  # 15GB
            'persistent': 10 * 1024 * 1024 * 1024, # 10GB
            'buffer': 9 * 1024 * 1024 * 1024      # 9GB
        }
        self.device_map = {}
        self.setup_memory_sharding()
    
    def setup_memory_sharding(self):
        """Set up memory sharding across available GPUs."""
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if num_devices > 1:
            # Distribute components across GPUs
            self.device_map = {
                'core': 0,  # Core module on first GPU
                'long_term': 1 if num_devices > 1 else 0,  # Long-term memory on second GPU if available
                'persistent': 2 if num_devices > 2 else 0,  # Persistent memory on third GPU if available
            }
        else:
            # Single GPU/CPU mode
            self.device_map = {
                'core': 0,
                'long_term': 0,
                'persistent': 0,
            }
    
    def get_device(self, component: str) -> torch.device:
        """Get the device for a specific component."""
        if torch.cuda.is_available():
            return torch.device(f'cuda:{self.device_map[component]}')
        return torch.device('cpu')
    
    def check_memory_usage(self, component: str, size_bytes: int) -> bool:
        """Check if adding a component would exceed VRAM budget."""
        if component not in self.component_budgets:
            raise ValueError(f"Unknown component: {component}")
        
        budget = self.component_budgets[component]
        # Check both component budget and total VRAM budget
        return (size_bytes <= budget and 
                (self.current_usage + size_bytes) <= self.vram_budget)
    
    def register_component(self, component: str, module: nn.Module):
        """Register a component and move it to the appropriate device."""
        # Calculate component size
        param_size = sum(getattr(p, 'real_numel', p.numel()) * p.element_size() for p in module.parameters())
        buffer_size = sum(getattr(b, 'real_numel', b.numel()) * b.element_size() for b in module.buffers())
        total_size = param_size + buffer_size
        
        # Check against budget
        if not self.check_memory_usage(component, total_size):
            raise RuntimeError(
                f"Component {component} exceeds VRAM budget. "
                f"Size: {total_size / 1024**3:.2f}GB, "
                f"Budget: {self.component_budgets[component] / 1024**3:.2f}GB"
            )
        
        # Move to appropriate device
        device = self.get_device(component)
        module.to(device)
        
        # Update current usage
        self.current_usage += total_size
        
    def get_total_usage(self) -> float:
        """Get total VRAM usage in GB."""
        return self.current_usage / (1024**3)
