"""
Tests for visualization utilities.
"""

import pytest
import os
import torch
import matplotlib.pyplot as plt
from llama_titan.utils.visualization import (
    plot_memory_usage,
    plot_memory_timeline,
    plot_component_distribution
)
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

@pytest.fixture
def component_budgets():
    return {
        'core': 30 * 1024**3,      # 30GB
        'long_term': 15 * 1024**3,  # 15GB
        'persistent': 10 * 1024**3, # 10GB
        'buffer': 9 * 1024**3      # 9GB
    }

def test_plot_memory_usage(component_budgets, tmp_path):
    save_path = os.path.join(tmp_path, 'memory_usage.png')
    plot_memory_usage(component_budgets, save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0

def test_plot_memory_timeline(component_budgets, tmp_path):
    save_path = os.path.join(tmp_path, 'memory_timeline.png')
    
    # Create sample usage history
    history = [
        {k: v * (1 + i * 0.1) for k, v in component_budgets.items()}
        for i in range(5)
    ]
    
    plot_memory_timeline(history, save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0

def test_plot_component_distribution(config, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    save_path = os.path.join(tmp_path, 'component_distribution.png')
    memory_manager = MemoryManager(config)
    
    plot_component_distribution(memory_manager, save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0

def test_visualization_style():
    """Test that plots use consistent style."""
    plt.style.use('default')  # Reset to default style
    component_budgets = {
        'core': 30 * 1024**3,
        'long_term': 15 * 1024**3,
        'persistent': 10 * 1024**3,
        'buffer': 9 * 1024**3
    }
    
    # Create and check plot (will be saved to memory)
    plot_memory_usage(component_budgets)
    
    # Verify plot properties
    fig = plt.gcf()
    ax = plt.gca()
    
    assert ax.get_ylabel() == 'Memory Usage (GB)'
    assert ax.get_title() == 'VRAM Usage by Component'
    assert ax.get_grid()
    
    plt.close()
