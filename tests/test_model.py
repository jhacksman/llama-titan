"""
Tests for the Titans Model implementation.
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

@pytest.fixture
def mock_embeddings():
    class MockEmbeddings(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        def forward(self, input_ids):
            return self.embedding(input_ids)
    return MockEmbeddings

def test_model_initialization(config):
    model = TitanModel(config)
    assert isinstance(model.short_term, nn.Module)
    assert isinstance(model.long_term, nn.Module)
    assert isinstance(model.persistent, nn.Module)
    assert isinstance(model.memory_manager, object)

def test_model_forward(config, mock_embeddings, monkeypatch):
    model = TitanModel(config)
    # Patch the get_input_embeddings method
    monkeypatch.setattr(model, "get_input_embeddings", lambda: mock_embeddings(config))
    
    batch_size = 2
    seq_length = 512
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    task_ids = torch.zeros(batch_size, dtype=torch.long)
    
    logits, surprise_scores, knowledge_attention = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task_ids=task_ids
    )
    
    assert logits.shape == (batch_size, seq_length, config.vocab_size)
    assert surprise_scores.shape == (batch_size, seq_length)
    assert knowledge_attention.shape == (batch_size, seq_length, config.knowledge_size)

def test_memory_integration(config, mock_embeddings, monkeypatch):
    model = TitanModel(config)
    monkeypatch.setattr(model, "get_input_embeddings", lambda: mock_embeddings(config))
    
    # Verify memory components are properly registered
    assert model.memory_manager.get_total_usage() > 0
    assert model.memory_manager.get_device('core') is not None
    assert model.memory_manager.get_device('long_term') is not None
    assert model.memory_manager.get_device('persistent') is not None

def test_vram_budget_compliance(config):
    model = TitanModel(config)
    total_usage = model.memory_manager.get_total_usage()
    # Ensure total VRAM usage is within 64GB budget
    assert total_usage < 64  # GB
