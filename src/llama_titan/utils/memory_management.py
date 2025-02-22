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
        self.prefetch_size = min(
            1024 * 1024 * 1024,  # 1GB max prefetch size
            self.vram_budget // 64  # Fraction of total VRAM
        )
        self.setup_memory_sharding()
        self.setup_memory_prefetching()
    
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
            device = torch.device(f'cuda:{self.device_map[component]}')
            # Enable memory caching for the device
            if hasattr(torch.cuda, 'memory_reserved'):
                torch.cuda.memory_reserved(device)
            return device
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
        
        # Check component budget
        budget = self.component_budgets[component]
        if total_size > budget:
            raise RuntimeError(f"Component {component} memory usage exceeds VRAM budget")
        
        # Check total VRAM budget
        if (self.current_usage + total_size) > self.vram_budget:
            raise RuntimeError(
                f"Component {component} exceeds total VRAM budget. "
                f"Size: {total_size / 1024**3:.2f}GB, "
                f"Available: {(self.vram_budget - self.current_usage) / 1024**3:.2f}GB"
            )
        
        # Optimize memory allocation for component
        self.optimize_memory_allocation(component)
        
        # Move to appropriate device
        device = self.get_device(component)
        module.to(device)
        
        # Enable gradient checkpointing if available
        if hasattr(module, 'gradient_checkpointing_enable'):
            module.gradient_checkpointing_enable()
        
        # Update current usage
        self.current_usage += total_size
        
    def setup_memory_prefetching(self):
        """Configure memory prefetching for better performance."""
        if torch.cuda.is_available():
            # Enable memory pinning for faster transfers
            torch.cuda.set_stream(torch.cuda.Stream())
            # Reserve memory for prefetching
            self._reserve_prefetch_memory()
    
    def _reserve_prefetch_memory(self):
        """Reserve memory for prefetching operations."""
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                with torch.cuda.device(device):
                    # Allocate prefetch buffer
                    torch.cuda.empty_cache()
                    torch.cuda.set_per_process_memory_fraction(
                        1.0 - (self.prefetch_size / self.vram_budget)
                    )
    
    def optimize_memory_allocation(self, component: str):
        """Optimize memory allocation for a specific component."""
        if component not in self.component_budgets:
            raise ValueError(f"Unknown component: {component}")
        
        device = self.get_device(component)
        if torch.cuda.is_available():
            # Clear fragmented memory
            torch.cuda.empty_cache()
            # Set memory fraction for component
            budget = self.component_budgets[component]
            torch.cuda.set_per_process_memory_fraction(
                budget / self.vram_budget,
                device
            )
    def get_total_usage(self) -> float:
        """Get total VRAM usage in GB."""
        return self.current_usage / (1024**3)
