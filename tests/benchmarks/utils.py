"""
Benchmarking utilities and shared imports.

This module handles import resolution for the benchmarking suite.
"""

import sys
from pathlib import Path

# Add the current directory to the path for relative imports
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

# Import all the main benchmarking components
from benchmark_entity_extractor import EntityBenchmarker, AnnotatedExample, BenchmarkResult
from local_conll_loader import load_conll2003_local, get_dataset_stats
from conll_data_loader import create_mini_conll_dataset

__all__ = [
    'EntityBenchmarker',
    'AnnotatedExample', 
    'BenchmarkResult',
    'load_conll2003_local',
    'get_dataset_stats',
    'create_mini_conll_dataset'
]