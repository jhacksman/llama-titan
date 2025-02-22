"""
Tests for the Long-term Memory implementation.
"""

import pytest
import torch
from llama_titan.memory.long_term import LongTermMemory

class TestConfig:
    def __init__(self):
        self.memory_size = 1024
        self.hidden_size = 4096
        self.surprise_threshold = 0.5
        self.momentum = 0.9

@pytest.fixture
def config():
    return TestConfig()

def test_long_term_memory_shape(config):
    batch_size = 2
    seq_length = 512
    
    model = LongTermMemory(config)
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
    output, surprise_scores = model(hidden_states)
    
    assert output.shape == (batch_size, seq_length, config.hidden_size)
    assert surprise_scores.shape == (batch_size, seq_length)

def test_surprise_computation(config):
    batch_size = 2
    seq_length = 512
    
    model = LongTermMemory(config)
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
    surprise_scores = model.compute_surprise(hidden_states)
    
    assert surprise_scores.shape == (batch_size, seq_length)
    assert (surprise_scores >= 0).all()  # Surprise scores should be non-negative

def test_memory_update(config):
    model = LongTermMemory(config)
    hidden_states = torch.randn(10, config.hidden_size)
    surprise_scores = torch.ones(10)
    
    old_memory = model.memory_bank.clone()
    model.update_memory(hidden_states, surprise_scores)
    
    assert not torch.equal(old_memory, model.memory_bank)
    assert model.memory_bank.shape == (config.memory_size, config.hidden_size)

def test_vram_usage(config):
    model = LongTermMemory(config)
    
    # Calculate approximate VRAM usage
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    # Ensure VRAM usage is within budget (15GB)
    assert total_size < 15 * 1024 * 1024 * 1024  # 15GB in bytes
