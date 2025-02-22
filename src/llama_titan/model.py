"""
Titans Model implementation integrating all memory components.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint

from .memory.core import ShortTermMemory
from .memory.long_term import LongTermMemory
from .memory.persistent import PersistentMemory
from .utils.memory_management import MemoryManager

class TitanModel(nn.Module):
    """Main model class integrating all memory components."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(config)
        
        # Initialize memory components
        self.short_term = ShortTermMemory(config)
        self.long_term = LongTermMemory(config)
        self.persistent = PersistentMemory(config)
        
        # Register components with memory manager
        self.memory_manager.register_component('core', self.short_term)
        self.memory_manager.register_component('long_term', self.long_term)
        self.memory_manager.register_component('persistent', self.persistent)
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get embeddings (assumed to be provided by LLaMA base)
        hidden_states = self.get_input_embeddings()(input_ids)
        
        # Process through short-term memory with gradient checkpointing during training
        if self.training:
            hidden_states, attention_weights = checkpoint(
                self.short_term,
                hidden_states,
                attention_mask,
                position_ids
            )
        else:
            hidden_states, attention_weights = self.short_term(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
        
        # Process through long-term memory
        hidden_states, surprise_scores = self.long_term(
            hidden_states,
            attention_mask=attention_mask
        )
        
        # Process through persistent memory
        if task_ids is None:
            task_ids = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        hidden_states, knowledge_attention = self.persistent(
            hidden_states,
            task_ids=task_ids,
            attention_mask=attention_mask
        )
        
        # Project to vocabulary
        logits = self.output_proj(hidden_states)
        
        return logits, surprise_scores, knowledge_attention
    
    def get_input_embeddings(self) -> nn.Module:
        """Returns the input embeddings module."""
        # This should be implemented by the LLaMA base model
        raise NotImplementedError("Input embeddings not implemented")
