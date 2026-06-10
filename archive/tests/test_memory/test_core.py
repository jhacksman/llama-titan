"""
Tests for the Core Module (Short-term Memory) implementation.
"""

import pytest
import torch
from llama_titan.memory.core import ShortTermMemory, SlidingWindowAttention

class TestConfig:
    def __init__(self):
        self.window_size = 512
        self.num_attention_heads = 32
        self.hidden_size = 4096

@pytest.fixture
def config():
    return TestConfig()

def test_sliding_window_attention_shape(config):
    batch_size = 2
    seq_length = 1024
    
    attention = SlidingWindowAttention(config)
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
    output, attention_weights = attention(hidden_states)
    
    assert output.shape == (batch_size, seq_length, config.hidden_size)
    assert attention_weights.shape == (batch_size, config.num_attention_heads, seq_length, seq_length)

def test_short_term_memory_forward(config):
    batch_size = 2
    seq_length = 1024
    
    model = ShortTermMemory(config)
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
    output, attention_weights = model(hidden_states)
    
    assert output.shape == (batch_size, seq_length, config.hidden_size)
    assert attention_weights.shape == (batch_size, config.num_attention_heads, seq_length, seq_length)

def test_sliding_window_mask(config):
    attention = SlidingWindowAttention(config)
    seq_length = 8
    window_size = 4
    
    mask = attention._create_sliding_window_mask(seq_length, window_size, torch.device('cpu'))
    
    # Check window size
    for i in range(seq_length):
        start = max(0, i - window_size // 2)
        end = min(seq_length, i + window_size // 2 + 1)
        assert mask[i, start:end].all()
        if start > 0:
            assert not mask[i, :start].any()
        if end < seq_length:
            assert not mask[i, end:].any()

def test_vram_usage(config):
    model = ShortTermMemory(config)
    
    # Calculate approximate VRAM usage
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    # Ensure VRAM usage is within budget (30GB)
    assert total_size < 30 * 1024 * 1024 * 1024  # 30GB in bytes
