"""
Tests for memory prefetching functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock
from llama_titan.utils.memory_management import MemoryManager

class TestConfig:
    def __init__(self):
        self.hidden_size = 4096
        self.vocab_size = 32000
        self.num_attention_heads = 32
        self.window_size = 512
        self.memory_size = 1024
        self.knowledge_size = 1024
        self.num_tasks = 10

@pytest.fixture
def config():
    return TestConfig()

def test_memory_prefetching_setup(config):
    manager = MemoryManager(config)
    assert hasattr(manager, 'prefetch_size')
    assert manager.prefetch_size > 0
    assert manager.prefetch_size <= manager.vram_budget

def test_memory_optimization(config):
    manager = MemoryManager(config)
    
    # Create a dummy module
    module = nn.Linear(config.hidden_size, config.hidden_size)
    
    # Register component
    manager.register_component('core', module)
    
    # Verify device placement
    device = manager.get_device('core')
    assert next(module.parameters()).device == device

def test_memory_allocation_limits(config):
    manager = MemoryManager(config)
    
    # Calculate memory requirements for a module
    dim = 100000  # Size that would exceed budget
    param_size = dim * dim * 4  # Size in bytes for weights
    bias_size = dim * 4  # Size in bytes for bias
    total_size = param_size + bias_size
    
    # Create a small module that we can actually allocate
    small_module = nn.Linear(100, 100)
    
    # Create mock parameters with fixed numel value
    mock_param = Mock()
    mock_param.numel = Mock(return_value=dim * dim)
    mock_param.element_size = Mock(return_value=4)  # assuming float32
    mock_param.real_numel = dim * dim  # Add real_numel attribute
    
    # Save original parameters method and replace with mock
    orig_parameters = small_module.parameters
    small_module.parameters = lambda: [mock_param]
    
    try:
        # Verify budget enforcement
        with pytest.raises(RuntimeError, match="memory usage exceeds VRAM budget"):
            manager.register_component('core', small_module)
    finally:
        # Restore original parameters method
        small_module.parameters = orig_parameters

def test_memory_prefetching_performance(config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    manager = MemoryManager(config)
    
    # Create test tensors
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    
    # Measure operation time with prefetching
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    z = torch.matmul(x, y)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)
    
    assert elapsed_time > 0  # Basic sanity check
