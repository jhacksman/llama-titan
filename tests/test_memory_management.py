"""
Tests for the Memory Management System.
"""

import pytest
import torch
import torch.nn as nn
from llama_titan.utils.memory_management import MemoryManager

class TestConfig:
    def __init__(self):
        pass

class DummyModule(nn.Module):
    def __init__(self, size_mb: int):
        super().__init__()
        # Create a parameter of specified size in MB
        num_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
        self.param = nn.Parameter(torch.randn(num_elements))

@pytest.fixture
def config():
    return TestConfig()

def test_memory_manager_initialization(config):
    manager = MemoryManager(config)
    assert manager.vram_budget == 64 * 1024 * 1024 * 1024  # 64GB
    assert manager.current_usage == 0
    assert len(manager.component_budgets) == 4  # core, long_term, persistent, buffer

def test_component_budgets(config):
    manager = MemoryManager(config)
    assert manager.component_budgets['core'] == 30 * 1024 * 1024 * 1024  # 30GB
    assert manager.component_budgets['long_term'] == 15 * 1024 * 1024 * 1024  # 15GB
    assert manager.component_budgets['persistent'] == 10 * 1024 * 1024 * 1024  # 10GB
    assert manager.component_budgets['buffer'] == 9 * 1024 * 1024 * 1024  # 9GB

def test_memory_usage_check(config):
    manager = MemoryManager(config)
    # Test with core module budget (30GB)
    assert manager.check_memory_usage('core', 29 * 1024 * 1024 * 1024)  # Should pass
    assert not manager.check_memory_usage('core', 31 * 1024 * 1024 * 1024)  # Should fail

def test_component_registration(config):
    manager = MemoryManager(config)
    # Create a 1GB dummy module
    module = DummyModule(1024)  # 1GB
    manager.register_component('core', module)
    assert manager.get_total_usage() < 30  # Should be around 1GB

def test_device_assignment(config):
    manager = MemoryManager(config)
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            assert manager.get_device('core') != manager.get_device('long_term')
    else:
        assert manager.get_device('core') == torch.device('cpu')

def test_memory_overflow_detection(config):
    manager = MemoryManager(config)
    # First register a module that uses a small amount of memory
    initial_module = DummyModule(1)  # 1MB
    manager.register_component('core', initial_module)
    
    # Now try to register another module that would exceed the budget
    # We'll simulate the size calculation instead of actually allocating memory
    class MockModule(nn.Module):
        def __init__(self):
            super().__init__()
            self._parameters = {'mock': torch.empty(1)}  # Minimal allocation
            
        def parameters(self):
            # Simulate having 31GB worth of parameters (8 bytes per float64)
            param = torch.nn.Parameter(torch.empty(1, dtype=torch.float64))
            param.real_numel = 31 * 1024 * 1024 * 1024 // 8  # Simulate 31GB
            yield param
            
        def buffers(self):
            return iter(())
            
    large_module = MockModule()
    with pytest.raises(RuntimeError, match=r".*exceeds VRAM budget.*"):
        manager.register_component('core', large_module)
