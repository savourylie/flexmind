# Development Guide

## Getting Started

1. **Setup Environment**
   ```bash
   cd flexmind
   uv sync
   ```

2. **Run Tests**
   ```bash
   uv run python -m pytest tests/ -v
   ```

3. **Run Examples**
   ```bash
   uv run python examples/demo.py
   ```

## Development Workflow

1. **TDD Approach**: Always write tests first
2. **Module Structure**: Keep modules focused and well-documented
3. **Import Consistency**: Use absolute imports from package root
4. **Type Hints**: Add type hints for all public APIs

## Testing Strategy

- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for full workflows
- Use pytest fixtures for common test setup
- Maintain >90% test coverage

## Future Architecture

```
flexmind/
├── chunking/        # ✅ Coreference-safe text chunking
├── knowledge_graph/ # 🔄 Entity-relationship extraction
├── memory/         # ⏳ Memory manager and retrieval
└── utils/          # ⏳ Shared utilities
```

## Code Style

- Follow PEP 8
- Use docstrings for all public methods
- Keep functions focused and testable
- Use meaningful variable names