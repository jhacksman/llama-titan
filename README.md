# LLaMA-Titan

Implementation of the Titans architecture, a memory-driven AI system based on LLaMA 7B 3.3. This architecture combines three distinct memory components:

1. Short-term Memory (Core Module)
2. Long-term Memory (Neural Module)
3. Persistent Memory (Task-specific Module)

## Architecture

The Titans architecture extends LLaMA with three memory components:

- **Core Module (Short-term Memory)**: Handles immediate context using sliding window attention
- **Long-term Memory**: Stores and retrieves historical data with surprise-based updates
- **Persistent Memory**: Encodes task-specific knowledge

## Installation

```bash
poetry install
```

## Development

```bash
# Run tests
poetry run pytest

# Format code
poetry run black .
```

## Memory Management

The system is designed to operate within a 64GB VRAM constraint:
- Core Module: ~30GB
- Long-term Memory: ~15GB
- Persistent Memory: ~10GB
- Buffer: ~9GB

## License

MIT
