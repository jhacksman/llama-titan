"""
Tests for gradient checkpointing functionality.
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

def test_gradient_checkpointing_training(config):
    model = TitanModel(config)
    model.train()  # Set to training mode
    
    # Create dummy inputs
    batch_size = 2
    seq_length = 512
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    task_ids = torch.zeros(batch_size, dtype=torch.long)
    
    # Forward pass with gradient checkpointing
    logits, surprise_scores, knowledge_attention = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task_ids=task_ids
    )
    
    # Verify outputs
    assert logits.shape == (batch_size, seq_length, config.vocab_size)
    assert surprise_scores.shape == (batch_size, seq_length)
    assert knowledge_attention.shape == (batch_size, seq_length, config.knowledge_size)

def test_gradient_checkpointing_inference(config):
    model = TitanModel(config)
    model.eval()  # Set to evaluation mode
    
    # Create dummy inputs
    batch_size = 2
    seq_length = 512
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    task_ids = torch.zeros(batch_size, dtype=torch.long)
    
    # Forward pass without gradient checkpointing
    with torch.no_grad():
        logits, surprise_scores, knowledge_attention = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_ids=task_ids
        )
    
    # Verify outputs
    assert logits.shape == (batch_size, seq_length, config.vocab_size)
    assert surprise_scores.shape == (batch_size, seq_length)
    assert knowledge_attention.shape == (batch_size, seq_length, config.knowledge_size)

def test_memory_efficiency(config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    model = TitanModel(config)
    model.train()
    model.cuda()
    
    # Create dummy inputs
    batch_size = 2
    seq_length = 512
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).cuda()
    attention_mask = torch.ones(batch_size, seq_length).cuda()
    task_ids = torch.zeros(batch_size, dtype=torch.long).cuda()
    
    # Record memory usage before forward pass
    torch.cuda.empty_cache()
    memory_before = torch.cuda.memory_allocated()
    
    # Forward pass with gradient checkpointing
    logits, surprise_scores, knowledge_attention = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task_ids=task_ids
    )
    
    # Record memory usage after forward pass
    memory_after = torch.cuda.memory_allocated()
    
    # Verify memory efficiency
    memory_increase = memory_after - memory_before
    assert memory_increase > 0  # Should use some memory
    assert memory_increase < model.memory_manager.vram_budget  # Should stay within budget
