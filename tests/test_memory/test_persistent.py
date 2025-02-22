"""
Tests for the Persistent Memory implementation.
"""

import pytest
import torch
from llama_titan.memory.persistent import PersistentMemory

class TestConfig:
    def __init__(self):
        self.hidden_size = 4096
        self.num_tasks = 10
        self.knowledge_size = 1024

@pytest.fixture
def config():
    return TestConfig()

def test_persistent_memory_shape(config):
    batch_size = 2
    seq_length = 512
    
    model = PersistentMemory(config)
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
    task_ids = torch.zeros(batch_size, dtype=torch.long)
    output, attention_probs = model(hidden_states, task_ids)
    
    assert output.shape == (batch_size, seq_length, config.hidden_size)
    assert attention_probs.shape == (batch_size, seq_length, config.knowledge_size)

def test_task_knowledge_retrieval(config):
    model = PersistentMemory(config)
    task_ids = torch.tensor([0, 1, 2])
    knowledge = model.get_task_knowledge(task_ids)
    
    assert knowledge.shape == (3, config.knowledge_size, config.hidden_size)
    assert not knowledge.requires_grad  # Knowledge bank should be static during inference

def test_different_tasks_different_knowledge(config):
    model = PersistentMemory(config)
    task_1 = torch.tensor([0])
    task_2 = torch.tensor([1])
    
    knowledge_1 = model.get_task_knowledge(task_1)
    knowledge_2 = model.get_task_knowledge(task_2)
    
    assert not torch.equal(knowledge_1, knowledge_2)

def test_vram_usage(config):
    model = PersistentMemory(config)
    
    # Calculate approximate VRAM usage
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    # Ensure VRAM usage is within budget (10GB)
    assert total_size < 10 * 1024 * 1024 * 1024  # 10GB in bytes
