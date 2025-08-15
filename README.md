# FlexMind 🧠

A TDD-driven knowledge graph memory system for LLMs with comprehensive benchmarking.

## Quick Start

```bash
# Install dependencies
uv sync

# Run interactive demos
uv run python demo_text_preprocessor.py
uv run python demo_entity_extractor.py

# Run benchmarks
uv run python run_benchmarks.py --quick
uv run python run_benchmarks.py --conll  # Full CoNLL-2003 evaluation
```

## Current Performance (CoNLL-2003 Test Set)

```
EntityExtractor Performance:
  F1-Score:   0.556 (honest baseline)
  Speed:      3297 tokens/sec
  
Strengths: Locations (92.3%), Persons (77.9%)
Bottlenecks: Organizations (11.1%), MISC entities (0%)
```

## Architecture

- **TextPreprocessor**: Dialog sliding windows + document chunking
- **EntityExtractor**: spaCy primary + DistilBERT fallback
- **Benchmarking**: Real CoNLL-2003 data (3,453 test sentences)
- **TDD Approach**: Comprehensive test coverage with performance tracking

## Core Philosophy

Traditional RAG systems use static chunking with photographic accuracy. FlexMind takes a different approach:

- **Adaptive over Static**: Memory chunks adjust based on context, not rigid token limits
- **Contextual Relevance**: Prioritizes meaningful relationships over perfect recall
- **Flexible Retrieval**: Knowledge graphs enable dynamic entity-relationship modifications
- **Cost-Conscious**: Efficient processing without expensive LLM calls for basic operations

## Documentation

- **[Quick Start Guide](QUICK_START.md)**: Usage examples and demos
- **[Project Structure](PROJECT_STRUCTURE.md)**: File organization and architecture  
- **[System Design](docs/SYSTEM_DESIGN.md)**: Detailed technical architecture
- **[Benchmarking Guide](BENCHMARKING_GUIDE.md)**: Performance evaluation system