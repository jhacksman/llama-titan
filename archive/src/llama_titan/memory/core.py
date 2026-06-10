"""
Core Module (Short-term Memory) implementation for the Titans architecture.
This module handles immediate context using a sliding window attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SlidingWindowAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window_size = config.window_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Create sliding window attention mask
        window_mask = self._create_sliding_window_mask(seq_length, self.window_size, hidden_states.device)
        if attention_mask is not None:
            # Convert attention_mask to boolean and reshape
            attention_mask = (attention_mask > 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_length]
            window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]
            window_mask = window_mask & attention_mask

        # Scaled dot-product attention with sliding window
        attention_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        attention_weights = attention_weights.masked_fill(~window_mask, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention_weights, value_states)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        output = self.o_proj(output)

        return output, attention_weights

    def _create_sliding_window_mask(self, seq_length: int, window_size: int, device: torch.device) -> torch.Tensor:
        """Creates a binary mask for sliding window attention."""
        mask = torch.zeros(seq_length, seq_length, device=device, dtype=torch.bool)
        for i in range(seq_length):
            start = max(0, i - window_size // 2)
            end = min(seq_length, i + window_size // 2 + 1)
            mask[i, start:end] = True
        return mask

class ShortTermMemory(nn.Module):
    """Core Module implementing short-term memory using sliding window attention."""
    
    def __init__(self, config):
        super().__init__()
        self.attention = SlidingWindowAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size)
        )
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.mlp_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with sliding window
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states, attention_weights = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attention_weights
