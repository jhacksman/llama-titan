"""
Persistent Memory implementation for the Titans architecture.
This module provides task-specific knowledge storage that remains static during inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

class PersistentMemory(nn.Module):
    """Task-specific knowledge storage that remains static during inference."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_tasks = config.num_tasks
        
        # Task-specific knowledge banks
        self.knowledge_bank = nn.Parameter(
            torch.randn(config.num_tasks, config.knowledge_size, config.hidden_size)
        )
        
        # Task embedding for knowledge retrieval
        self.task_embedding = nn.Embedding(config.num_tasks, config.hidden_size)
        
        # Neural networks for knowledge access
        self.query_net = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_net = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_net = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.output_layer = nn.Linear(2 * config.hidden_size, config.hidden_size)
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def get_task_knowledge(self, task_ids: torch.Tensor) -> torch.Tensor:
        """Retrieve task-specific knowledge based on task IDs."""
        with torch.no_grad():
            task_emb = self.task_embedding(task_ids)
            return self.knowledge_bank[task_ids].detach()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        task_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Get task-specific knowledge
        task_knowledge = self.get_task_knowledge(task_ids)
        
        # Compute queries, keys, and values
        queries = self.query_net(hidden_states)
        knowledge_keys = self.key_net(task_knowledge)
        knowledge_values = self.value_net(task_knowledge)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, knowledge_keys.transpose(-2, -1))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        
        if attention_mask is not None:
            # Convert attention_mask to boolean and reshape
            attention_mask = (attention_mask > 0).unsqueeze(-1)
            attention_scores = attention_scores.masked_fill(~attention_mask, float('-inf'))
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # Retrieve from knowledge bank
        knowledge_output = torch.matmul(attention_probs, knowledge_values)
        
        # Combine current hidden states with knowledge output
        combined = torch.cat([hidden_states, knowledge_output], dim=-1)
        output = self.output_layer(combined)
        
        # Apply layer normalization
        output = self.layer_norm(output + hidden_states)
        
        return output, attention_probs
