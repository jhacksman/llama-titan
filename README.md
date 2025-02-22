# LLaMA-Titan

Implementation of the Titans architecture, a memory-driven AI system based on LLaMA 7B 3.3. This architecture combines three distinct memory components to enhance context handling and task-specific knowledge retention while maintaining efficient memory usage.

## Architecture Overview

The Titans architecture extends LLaMA with three specialized memory components:

### 1. Core Module (Short-term Memory)
- Handles immediate context processing using sliding window attention
- Processes sequences up to 2048 tokens
- Implements gradient checkpointing for memory efficiency
- Uses mixed precision (FP16/BF16) for computation
- VRAM Budget: 30GB

### 2. Long-term Memory
- Maintains historical context through neural storage
- Implements surprise-based update mechanism
- Features dynamic memory bank management
- Optimizes retrieval through prefetching
- VRAM Budget: 15GB

### 3. Persistent Memory
- Stores task-specific knowledge
- Enables specialized information retention
- Supports multiple task contexts
- Implements efficient knowledge retrieval
- VRAM Budget: 10GB

## Memory Management

The system operates within a 64GB VRAM constraint with careful budget allocation:

```
Total VRAM: 64GB
├── Core Module: 30GB
├── Long-term Memory: 15GB
├── Persistent Memory: 10GB
└── Buffer: 9GB
```

### Memory Optimization Features
- Memory sharding across multiple GPUs
- Dynamic memory prefetching
- Gradient checkpointing support
- Mixed precision training
- Automatic memory cleanup

## Installation

```bash
# Install dependencies
poetry install

# Optional: Install development dependencies
poetry install --with dev
```

## Usage

```python
from llama_titan import TitanModel

# Initialize model
model = TitanModel(config)

# Forward pass with all memory components
logits, surprise_scores, knowledge_attention = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    task_ids=task_ids
)
```

## Development

```bash
# Run tests
poetry run pytest

# Run specific test suite
poetry run pytest tests/test_memory/

# Format code
poetry run black .

# Run type checks
poetry run mypy .
```

## Performance Guidelines

1. Memory Usage
   - Monitor VRAM usage with visualization tools
   - Use memory profiling during development
   - Enable gradient checkpointing for large batches

2. Optimization
   - Enable mixed precision training
   - Utilize memory prefetching
   - Implement proper cleanup in custom modules

3. Multi-GPU Support
   - Configure device mapping for components
   - Enable memory sharding when available
   - Optimize cross-device communication

## API Documentation

### TitanModel

```python
class TitanModel:
    """Main model integrating all memory components."""
    
    def forward(
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            position_ids: Optional position IDs
            task_ids: Task identifiers for persistent memory
            
        Returns:
            logits: Output logits
            surprise_scores: Memory update indicators
            knowledge_attention: Task-specific attention
        """
```

### Memory Components

```python
class ShortTermMemory:
    """Core module for immediate context processing."""
    
    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]

class LongTermMemory:
    """Neural system for historical context."""
    
    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]

class PersistentMemory:
    """Task-specific knowledge storage."""
    
    def forward(
        hidden_states: torch.Tensor,
        task_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]
```

## License

MIT
