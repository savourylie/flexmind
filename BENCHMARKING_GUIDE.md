# FlexMind Benchmarking System 🎯

## Overview

A comprehensive benchmarking system to measure and track EntityExtractor performance. This system provides **objective metrics**, **regression detection**, and **bottleneck identification** for data-driven optimization.

## Why This Matters

✅ **Establish Baselines**: Know current performance before optimizing  
✅ **Detect Regressions**: Catch performance drops when making changes  
✅ **Identify Bottlenecks**: Find the weakest component in the pipeline  
✅ **Measure Improvements**: Quantify gains from optimizations  
✅ **Debug Issues**: Isolate problems to specific components  

## Current Performance Baseline

### EntityExtractor Metrics (as of current implementation):
```
Overall Performance:
  F1-Score:   0.839  (83.9% - Good quality)
  Precision:  0.867  (86.7% - High accuracy)  
  Recall:     0.812  (81.2% - Good coverage)
  Speed:      245-2558 tokens/sec (depends on fallback usage)

Per Entity Type:
  GPE (locations):  F1=1.000 (Perfect)
  DATE:            F1=1.000 (Perfect) 
  MONEY:           F1=1.000 (Perfect)
  PERSON:          F1=0.800 (Very good)
  ORG:             F1=0.571 (Needs improvement)

Configuration Impact:
  SpaCy Only:      F1=0.839, Speed=2558 tok/s (Fastest)
  Hybrid:          F1=0.839, Speed=145 tok/s  (Balanced)
  High Confidence: F1=0.839, Speed=869 tok/s  (Conservative)
```

## Quick Usage

### Run Benchmarks

```bash
# Quick development checkpoint (30 seconds)
uv run python run_benchmarks.py --quick

# Full benchmark suite (2-3 minutes)
uv run python run_benchmarks.py --full

# Check for performance regression
uv run python run_benchmarks.py --regression  

# Compare different configurations
uv run python run_benchmarks.py --compare
```

### Pytest Integration

```bash
# Run benchmark tests only
uv run pytest tests/benchmarks/ -v --benchmark-only

# Run specific benchmark
uv run pytest tests/benchmarks/test_entity_benchmarks.py::test_baseline_benchmark -v

# Full test suite including benchmarks
uv run pytest tests/ -v
```

## Benchmark Components

### 1. EntityBenchmarker
- **Precision/Recall/F1**: Standard NER evaluation metrics
- **Speed**: Tokens processed per second
- **Fallback Rate**: How often DistilBERT is triggered
- **Per-Entity Analysis**: Performance breakdown by entity type

### 2. Standard Datasets
- **Sample Data**: Hand-crafted examples for our use case
- **Dialog Data**: Conversational text scenarios  
- **CoNLL Format**: Standard NER benchmark format
- **Mini CoNLL**: Realistic tech/business entities

### 3. Configuration Testing
- **SpaCy Only**: Fast but limited coverage
- **Hybrid**: Balance of speed and coverage
- **Confidence Variants**: High/low threshold comparison
- **Custom Configs**: Test specific optimizations

## Key Insights from Benchmarks

### 🎯 **Strengths**
- **Location/Date/Money**: Perfect extraction (F1=1.0)
- **Person Names**: Very good performance (F1=0.8) 
- **Speed**: Excellent with spaCy-only (2500+ tok/s)
- **Fallback Strategy**: Works correctly (20% activation)

### ⚠️ **Bottlenecks**
- **Organizations**: Moderate performance (F1=0.57)
  - Likely issue: Modern tech companies not in spaCy training data
  - Solution: Improve fallback, add custom patterns
- **Hybrid Speed**: Slow when fallback triggers (145 tok/s)
  - Solution: Optimize fallback conditions

### 🚀 **Recommendations**
1. **For Speed**: Use SpaCy-only configuration (2558 tok/s, same quality)
2. **For Coverage**: Improve ORG entity detection patterns
3. **For Production**: Monitor fallback rate (<30% is healthy)

## File Structure

```
tests/benchmarks/
├── __init__.py
├── benchmark_entity_extractor.py    # Core benchmarking framework
├── test_entity_benchmarks.py        # Pytest integration  
└── conll_data_loader.py             # Standard dataset support

benchmark_results/
├── quick_checkpoint.json            # Development checkpoints
├── baselines/                       # Regression testing baselines
└── full_suite/                      # Complete benchmark results

run_benchmarks.py                    # Easy CLI interface
```

## Adding New Benchmarks

### 1. Custom Dataset
```python
from tests.benchmarks.benchmark_entity_extractor import AnnotatedExample

custom_data = [
    AnnotatedExample(
        text="Your custom text here",
        entities=[("entity_text", "ENTITY_TYPE", start_pos, end_pos)]
    )
]

benchmarker = EntityBenchmarker()
result = benchmarker.benchmark_on_dataset(custom_data)
```

### 2. New Configuration
```python
custom_extractor = EntityExtractor(
    confidence_threshold=0.8,
    use_fallback=True
    # Add your custom parameters
)

benchmarker = EntityBenchmarker(custom_extractor)
result = benchmarker.benchmark_on_dataset(test_data)
```

### 3. CoNLL Format Data
```python
from tests.benchmarks.conll_data_loader import load_conll_file

# Load standard CoNLL-2003 format file
examples = load_conll_file(Path("path/to/conll/file.txt"))
result = benchmarker.benchmark_on_dataset(examples)
```

## Regression Testing

The system automatically saves baselines and compares new runs:

```bash
# First run establishes baseline
uv run python run_benchmarks.py --quick

# Later runs check for regression
uv run python run_benchmarks.py --regression
# ✅ No significant regression detected
# OR
# ⚠️ WARNING: F1-score regression detected!
```

## Integration with Development Workflow

### 1. **Pre-commit Checks**
```bash
# Add to git hooks or CI
uv run python run_benchmarks.py --regression
```

### 2. **Development Checkpoints**  
```bash
# After implementing changes
uv run python run_benchmarks.py --quick
```

### 3. **Release Testing**
```bash
# Before releases
uv run python run_benchmarks.py --full
```

## Next Steps

1. **Improve ORG Detection**: Focus optimization on organization entities
2. **Add More Datasets**: Include domain-specific benchmarks
3. **Speed Optimization**: Reduce fallback overhead
4. **Continuous Integration**: Automate benchmark runs
5. **A/B Testing**: Compare multiple extraction strategies

## Performance Targets

**Current**: F1=0.839, Speed=245-2558 tok/s  
**Target**: F1>0.85, Speed>1000 tok/s consistently  
**Stretch**: F1>0.90, Speed>5000 tok/s  

This benchmarking system gives us the **data-driven foundation** to optimize systematically and catch regressions early! 🚀