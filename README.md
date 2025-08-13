# FlexMind: Human-like Memory Mechanism

A memory system inspired by "Why We Remember" that prioritizes **flexibility and contextual adaptation** over static, photographic accuracy.

## 🧠 Core Philosophy

Traditional RAG systems use static chunking with photographic accuracy. FlexMind takes a different approach:

- **Adaptive over Static**: Memory chunks adjust based on context, not rigid token limits
- **Contextual Relevance**: Prioritizes maintaining meaningful relationships over perfect recall
- **Flexible Retrieval**: Knowledge graphs enable dynamic entity-relationship modifications
- **Cost-Conscious**: Efficient processing without expensive LLM calls for basic operations

## 🏗️ Project Structure

```
flexmind/
├── flexmind/                 # Main package
│   ├── chunking/            # Coreference-safe chunking
│   │   └── coref_chunker.py # Human-like text chunking
│   ├── knowledge_graph/     # Entity-relationship extraction (WIP)
│   ├── memory/             # Core memory management (WIP)
│   └── utils/              # Shared utilities (WIP)
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── examples/               # Usage examples and demos
├── docs/                   # Documentation
└── CLAUDE.md              # Project guidelines and vision
```

## ✨ Features

### 🎯 Coreference-Safe Chunking
- **Start Rule**: Prevents chunks from starting with orphaned pronouns
- **Anaphora Hazard Detection**: Expands context for high pronoun density (≥15%)
- **Sentence Start Rule**: Handles He/She/They/This/That/It sentence beginnings
- **Fallback Strategies**: Uses concrete nouns when named entities aren't available
- **Flexible Boundaries**: Respects sentence/paragraph boundaries while maintaining coherence

### 🚀 Coming Soon
- **Knowledge Graph Construction**: Dynamic entity-relationship mapping
- **Memory Manager**: Integrated retrieval and storage system  
- **Dialogue/Chat Chunking**: Specialized handling for conversational text
- **Event Timeline Processing**: Time-aware memory organization

## 🚀 Quick Start

```python
from flexmind import CorefSafeChunker

# Create a human-like chunker
chunker = CorefSafeChunker(target_size=300, max_size=500)

# Chunk text with coreference safety
text = "Dr. Sarah Martinez led the research team. She discovered breakthrough results."
chunks = chunker.chunk(text)

for chunk in chunks:
    print(f"Text: {chunk.text}")
    print(f"Tokens: {chunk.metadata['token_count']}")
    print(f"Has entities: {chunk.metadata['has_entities']}")
    print(f"Pronoun density: {chunk.metadata['pronoun_density']:.1%}")
```

## 🧪 Testing

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run specific test suites
uv run python -m pytest tests/unit/ -v
uv run python -m pytest tests/integration/ -v

# Run examples
uv run python examples/demo.py
```

## 📚 Examples

- `examples/demo.py` - Comprehensive demonstration of coreference-safe features
- More examples coming as we build knowledge graph and memory components

## 🎯 Roadmap

1. ✅ **Phase 1**: Coreference-safe chunking foundation
2. 🔄 **Phase 2**: Knowledge graph entity-relationship extraction
3. ⏳ **Phase 3**: Integrated memory manager with flexible retrieval
4. ⏳ **Phase 4**: Dialogue and event-specific processing
5. ⏳ **Phase 5**: Evaluation metrics and performance optimization

## 🤝 Development

FlexMind uses:
- **Python 3.11+** with uv for dependency management
- **spaCy** for NLP processing (efficient, no LLM calls)
- **pytest** for testing
- **TDD approach** for reliable development

See `CLAUDE.md` for detailed development guidelines and project vision.

## 📄 License

[License to be added]