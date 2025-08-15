# FlexMind Project Structure

## Overview
A TDD-driven knowledge graph memory system with comprehensive benchmarking using real CoNLL-2003 data.

## Directory Structure

```
flexmind/
├── flexmind/                    # Core implementation
│   ├── core/
│   │   ├── preprocessing/
│   │   │   └── text.py         # TextPreprocessor with TextChunk objects
│   │   ├── extractors/
│   │   │   └── entities.py     # EntityExtractor with Entity objects
│   │   └── storage/            # Future: Neo4j, ChromaDB integration
│   ├── api/                    # Future: FastAPI endpoints
│   └── scripts/                # Utility scripts
│
├── tests/
│   ├── unit/                   # Component unit tests
│   │   ├── test_text_preprocessor.py
│   │   └── test_entity_extractor.py
│   ├── benchmarks/             # Performance benchmarking
│   │   ├── utils.py           # Import utilities
│   │   ├── benchmark_entity_extractor.py  # Core benchmarking framework
│   │   ├── local_conll_loader.py         # CoNLL-2003 dataset loader
│   │   ├── full_conll_benchmark.py       # Full dataset benchmarking
│   │   └── test_entity_benchmarks.py     # Pytest integration
│   ├── integration/           # Cross-component tests
│   └── performance/           # Performance regression tests
│
├── data/
│   └── colnn2003/            # Local CoNLL-2003 parquet files
│       ├── train.parquet     # 14,041 sentences
│       ├── dev.parquet       # 3,250 sentences
│       └── test.parquet      # 3,453 sentences
│
├── benchmark_results/         # Saved benchmark outputs
│   ├── full_conll2003/       # Full dataset results
│   └── quick_checkpoint.json # Development checkpoints
│
├── docs/
│   └── SYSTEM_DESIGN.md      # Detailed architecture
│
├── demo_text_preprocessor.py  # Interactive demos
├── demo_entity_extractor.py   # Interactive demos
└── run_benchmarks.py          # CLI benchmark runner
```

## Key Components

### 1. Core Implementation
- **TextPreprocessor**: Handles dialog/document chunking with readable TextChunk objects
- **EntityExtractor**: spaCy + DistilBERT hybrid NER with Entity objects
- **Clear String Representations**: No more confusing tuples/dictionaries

### 2. Benchmarking System
- **Real CoNLL-2003 Data**: 3,453 test sentences vs 5-example toy data
- **Comprehensive Metrics**: Precision/Recall/F1 + per-entity analysis
- **Performance Tracking**: Speed benchmarking + regression detection
- **SOTA Comparison**: Compare against research literature

### 3. Development Tools
- **Interactive Demos**: Test components with your own text
- **CLI Benchmarking**: Quick checkpoints vs full evaluations
- **TDD Integration**: Pytest + benchmarking integration

## Current Performance (CoNLL-2003)

```
EntityExtractor Benchmark Results:
  F1-Score:   0.556 (honest baseline)
  Precision:  0.466 (moderate accuracy)
  Recall:     0.690 (good coverage)
  Speed:      3297 tokens/sec (spaCy-only)

Per-Entity Performance:
  Locations:  F1=0.923 (excellent)
  Persons:    F1=0.779 (very good)  
  Orgs:       F1=0.111 (major bottleneck)
  Misc:       F1=0.000 (not detected)
```

## Usage Commands

```bash
# Quick development check
uv run python run_benchmarks.py --quick

# Real CoNLL-2003 benchmark
uv run python run_benchmarks.py --conll

# Interactive demos
uv run python demo_text_preprocessor.py
uv run python demo_entity_extractor.py

# Run all tests
uv run pytest tests/ -v

# Benchmark-only tests  
uv run pytest tests/benchmarks/ -v --benchmark-only
```

## Development Workflow

1. **Make Changes**: Modify core components
2. **Run Tests**: `uv run pytest tests/unit/ -v`
3. **Quick Benchmark**: `uv run python run_benchmarks.py --quick`
4. **Full Evaluation**: `uv run python run_benchmarks.py --conll`
5. **Commit**: Only after tests pass and no regression

## Next Steps

1. **Optimize ORG Detection**: Primary bottleneck (F1=0.11)
2. **Add MISC Entity Support**: Currently F1=0.0  
3. **Implement RelationExtractor**: Extract (subject, predicate, object) triples
4. **Neo4j Integration**: Knowledge graph storage
5. **Vector Store**: ChromaDB for hybrid retrieval

## File Cleanup Status

✅ **Removed**:
- `debug_entities.py` (temporary debug file)
- `test_string_representations.py` (temporary demo)
- `tests/benchmarks/huggingface_loader.py` (unused)
- Unused legacy dependencies

✅ **Organized**:
- Clean import structure via `tests/benchmarks/utils.py`
- Consolidated documentation
- Clear project structure

✅ **Maintained**:
- All functional code and tests
- Benchmark results and data
- Interactive demos
- Core implementation