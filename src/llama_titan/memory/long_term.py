"""
Long-term Memory implementation for the Titans architecture.
This module handles historical data storage and retrieval with surprise-based updates.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

class LongTermMemory(nn.Module):
    """Neural memory system for storing and retrieving historical data."""
    
    def __init__(self, config):
        super().__init__()
        self.memory_size = config.memory_size
        self.hidden_size = config.hidden_size
        self.surprise_threshold = config.surprise_threshold
        self.momentum = config.momentum
        
        # Memory bank for storing historical information
        self.memory_bank = nn.Parameter(torch.randn(config.memory_size, config.hidden_size))
        self.memory_importance = nn.Parameter(torch.ones(config.memory_size))
        
        # Neural networks for memory operations
        self.query_net = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_net = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_net = nn.Linear(config.hidden_size, config.hidden_size)
        self.surprise_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1)
        )
        
        self.output_layer = nn.Linear(2 * config.hidden_size, config.hidden_size)
        
    def compute_surprise(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute surprise scores for input states."""
        # Apply ReLU to ensure non-negative surprise scores
        return torch.relu(self.surprise_net(hidden_states)).squeeze(-1)
    
    def update_memory(self, hidden_states: torch.Tensor, surprise_scores: torch.Tensor):
        """Update memory bank based on surprise scores."""
        # Find least important memories to replace
        _, indices = torch.topk(self.memory_importance, k=surprise_scores.size(0), largest=False)
        
        # Update memory bank
        self.memory_bank.data[indices] = hidden_states
        
        # Update importance scores with momentum
        new_importance = surprise_scores.clone()
        self.memory_importance.data[indices] = (
            self.momentum * self.memory_importance[indices] +
            (1 - self.momentum) * new_importance
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Compute queries, keys, and values
        queries = self.query_net(hidden_states)
        memory_keys = self.key_net(self.memory_bank)
        memory_values = self.value_net(self.memory_bank)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, memory_keys.t())
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        
        if attention_mask is not None:
            # Convert attention_mask to boolean and reshape
            attention_mask = (attention_mask > 0).unsqueeze(-1)
            attention_scores = attention_scores.masked_fill(~attention_mask, float('-inf'))
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # Retrieve from memory
        memory_output = torch.matmul(attention_probs, memory_values)
        
        # Compute surprise scores
        surprise_scores = self.compute_surprise(hidden_states)
        
        # Update memory for highly surprising inputs
        surprising_mask = surprise_scores > self.surprise_threshold
        if surprising_mask.any():
            surprising_states = hidden_states[surprising_mask]
            surprising_scores = surprise_scores[surprising_mask]
            self.update_memory(surprising_states, surprising_scores)
        
        # Combine current hidden states with memory output
        combined = torch.cat([hidden_states, memory_output], dim=-1)
        output = self.output_layer(combined)
        
        return output, surprise_scores
