"""
Tests for mixed precision functionality.
"""

import pytest
import torch
import torch.nn as nn
from llama_titan.model import TitanModel

class TestConfig:
    def __init__(self):
        self.hidden_size = 4096
        self.vocab_size = 32000
        self.num_attention_heads = 32
        self.window_size = 512
        self.memory_size = 1024
        self.knowledge_size = 1024
        self.num_tasks = 10
        self.surprise_threshold = 0.5
        self.momentum = 0.9

@pytest.fixture
def config():
    return TestConfig()

def test_mixed_precision_output_types(config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    model = TitanModel(config)
    model.cuda()
    
    # Create dummy inputs
    batch_size = 2
    seq_length = 512
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).cuda()
    attention_mask = torch.ones(batch_size, seq_length).cuda()
    task_ids = torch.zeros(batch_size, dtype=torch.long).cuda()
    
    # Forward pass with mixed precision
    with torch.cuda.amp.autocast():
        logits, surprise_scores, knowledge_attention = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_ids=task_ids
        )
    
    # Verify output types
    assert logits.dtype == torch.float16
    assert surprise_scores.dtype == torch.float16
    assert knowledge_attention.dtype == torch.float16

def test_mixed_precision_memory_efficiency(config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    model = TitanModel(config)
    model.cuda()
    
    # Create dummy inputs
    batch_size = 2
    seq_length = 512
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).cuda()
    attention_mask = torch.ones(batch_size, seq_length).cuda()
    task_ids = torch.zeros(batch_size, dtype=torch.long).cuda()
    
    # Record memory usage with FP32
    torch.cuda.empty_cache()
    memory_before_fp32 = torch.cuda.memory_allocated()
    
    # Forward pass with FP32
    with torch.cuda.amp.autocast(enabled=False):
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_ids=task_ids
        )
    
    memory_after_fp32 = torch.cuda.memory_allocated()
    fp32_memory = memory_after_fp32 - memory_before_fp32
    
    # Reset memory state
    del _
    torch.cuda.empty_cache()
    
    # Record memory usage with mixed precision
    memory_before_fp16 = torch.cuda.memory_allocated()
    
    # Forward pass with mixed precision
    with torch.cuda.amp.autocast():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_ids=task_ids
        )
    
    memory_after_fp16 = torch.cuda.memory_allocated()
    fp16_memory = memory_after_fp16 - memory_before_fp16
    
    # Verify memory efficiency
    assert fp16_memory < fp32_memory  # Mixed precision should use less memory

def test_mixed_precision_numerical_stability(config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    model = TitanModel(config)
    model.cuda()
    
    # Create dummy inputs
    batch_size = 2
    seq_length = 512
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).cuda()
    attention_mask = torch.ones(batch_size, seq_length).cuda()
    task_ids = torch.zeros(batch_size, dtype=torch.long).cuda()
    
    # Forward pass with mixed precision
    with torch.cuda.amp.autocast():
        logits, surprise_scores, knowledge_attention = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_ids=task_ids
        )
    
    # Check for NaN values
    assert not torch.isnan(logits).any()
    assert not torch.isnan(surprise_scores).any()
    assert not torch.isnan(knowledge_attention).any()
    
    # Check for infinity values
    assert not torch.isinf(logits).any()
    assert not torch.isinf(surprise_scores).any()
    assert not torch.isinf(knowledge_attention).any()
