#!/usr/bin/env python3
"""
FlexMind Benchmarking Suite Runner

This script provides easy commands to run comprehensive benchmarks on our
EntityExtractor and other components. Use this to establish baselines,
check for regressions, and measure performance improvements.

Usage:
  uv run python run_benchmarks.py --quick          # Quick baseline check
  uv run python run_benchmarks.py --full           # Full benchmark suite  
  uv run python run_benchmarks.py --regression     # Check against baseline
  uv run python run_benchmarks.py --compare        # Compare configurations
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from flexmind.core.extractors.entities import EntityExtractor
from tests.benchmarks.benchmark_entity_extractor import (
    EntityBenchmarker, 
    create_sample_benchmark_data, 
    create_dialog_benchmark_data
)
from tests.benchmarks.conll_data_loader import create_mini_conll_dataset


def quick_benchmark() -> Dict[str, Any]:
    """Run a quick benchmark for development feedback."""
    print("=" * 60)
    print("QUICK BENCHMARK - Development Checkpoint")
    print("=" * 60)
    
    benchmarker = EntityBenchmarker()
    sample_data = create_sample_benchmark_data()
    
    result = benchmarker.benchmark_on_dataset(sample_data)
    benchmarker.print_results(result)
    
    # Save as quick checkpoint
    checkpoint_path = Path("benchmark_results/quick_checkpoint.json")
    benchmarker.save_results(result, checkpoint_path)
    print(f"\nSaved checkpoint to: {checkpoint_path}")
    
    return {
        'f1': result.f1,
        'precision': result.precision,
        'recall': result.recall,
        'speed': result.speed_tokens_per_sec
    }


def full_benchmark_suite():
    """Run comprehensive benchmark suite."""
    print("=" * 60)
    print("FULL BENCHMARK SUITE")
    print("=" * 60)
    
    results = {}
    
    # 1. Sample data benchmark
    print("\n1. Sample Data Benchmark")
    print("-" * 30)
    benchmarker = EntityBenchmarker()
    sample_data = create_sample_benchmark_data()
    sample_result = benchmarker.benchmark_on_dataset(sample_data)
    benchmarker.print_results(sample_result)
    results['sample'] = sample_result
    
    # 2. Dialog data benchmark  
    print("\n2. Dialog Data Benchmark")
    print("-" * 30)
    dialog_data = create_dialog_benchmark_data()
    dialog_result = benchmarker.benchmark_on_dataset(dialog_data)
    benchmarker.print_results(dialog_result)
    results['dialog'] = dialog_result
    
    # 3. CoNLL-style data benchmark
    print("\n3. CoNLL-Style Data Benchmark")
    print("-" * 30)
    conll_data = create_mini_conll_dataset()
    conll_result = benchmarker.benchmark_on_dataset(conll_data)
    benchmarker.print_results(conll_result)
    results['conll'] = conll_result
    
    # 4. Configuration comparison
    print("\n4. Configuration Comparison")
    print("-" * 30)
    configs = {
        'spacy_only': EntityExtractor(use_fallback=False),
        'hybrid': EntityExtractor(use_fallback=True),
        'high_confidence': EntityExtractor(confidence_threshold=0.9),
        'low_confidence': EntityExtractor(confidence_threshold=0.5)
    }
    
    config_results = {}
    for name, extractor in configs.items():
        print(f"\n  Testing {name}:")
        config_benchmarker = EntityBenchmarker(extractor)
        config_result = config_benchmarker.benchmark_on_dataset(sample_data)
        config_results[name] = config_result
        print(f"    F1: {config_result.f1:.3f} | Speed: {config_result.speed_tokens_per_sec:.0f} tok/s | Fallback: {config_result.fallback_rate:.1%}")
    
    results['configurations'] = config_results
    
    # Save all results
    results_dir = Path("benchmark_results/full_suite")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    benchmarker.save_results(sample_result, results_dir / "sample_data.json")
    benchmarker.save_results(dialog_result, results_dir / "dialog_data.json") 
    benchmarker.save_results(conll_result, results_dir / "conll_data.json")
    
    for name, result in config_results.items():
        EntityBenchmarker(configs[name]).save_results(result, results_dir / f"config_{name}.json")
    
    print(f"\nAll results saved to: {results_dir}")
    return results


def regression_check():
    """Check current performance against saved baselines."""
    print("=" * 60)
    print("REGRESSION CHECK")
    print("=" * 60)
    
    baseline_dir = Path("benchmark_results/baselines")
    baseline_files = {
        'sample': baseline_dir / "entity_extractor_baseline.json",
        'dialog': baseline_dir / "dialog_baseline.json"
    }
    
    benchmarker = EntityBenchmarker()
    
    # Check sample data regression
    if baseline_files['sample'].exists():
        print("\nChecking sample data regression...")
        sample_data = create_sample_benchmark_data()
        current_result = benchmarker.benchmark_on_dataset(sample_data)
        comparison = benchmarker.compare_with_baseline(current_result, baseline_files['sample'])
        
        # Alert on significant regression
        if comparison.get('f1_change', 0) < -0.05:
            print("⚠️  WARNING: F1-score regression detected!")
            return False
        if comparison.get('speed_change', 0) < -1000:
            print("⚠️  WARNING: Speed regression detected!")
            return False
    else:
        print("No baseline found - run full benchmark to establish baseline")
        return None
    
    print("✅ No significant regression detected")
    return True


def compare_configurations():
    """Compare different EntityExtractor configurations."""
    print("=" * 60)
    print("CONFIGURATION COMPARISON")
    print("=" * 60)
    
    # Define configurations to compare
    configs = {
        'Default': EntityExtractor(),
        'SpaCy Only': EntityExtractor(use_fallback=False),
        'High Confidence': EntityExtractor(confidence_threshold=0.9),
        'Low Confidence': EntityExtractor(confidence_threshold=0.5),
        'Fast Setup': EntityExtractor(use_fallback=False, confidence_threshold=0.8)
    }
    
    sample_data = create_sample_benchmark_data()
    
    print(f"{'Configuration':<15} {'F1':<6} {'Prec':<6} {'Rec':<6} {'Speed':<8} {'Fallback':<8}")
    print("-" * 60)
    
    results = {}
    for name, extractor in configs.items():
        benchmarker = EntityBenchmarker(extractor)
        result = benchmarker.benchmark_on_dataset(sample_data)
        
        print(f"{name:<15} {result.f1:<6.3f} {result.precision:<6.3f} {result.recall:<6.3f} {result.speed_tokens_per_sec:<8.0f} {result.fallback_rate:<8.1%}")
        results[name] = result
    
    # Find best configuration
    best_f1 = max(results.items(), key=lambda x: x[1].f1)
    best_speed = max(results.items(), key=lambda x: x[1].speed_tokens_per_sec)
    
    print(f"\nBest F1-score: {best_f1[0]} ({best_f1[1].f1:.3f})")
    print(f"Fastest: {best_speed[0]} ({best_speed[1].speed_tokens_per_sec:.0f} tokens/sec)")
    
    return results


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="FlexMind Benchmarking Suite")
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark')
    parser.add_argument('--full', action='store_true', help='Run full benchmark suite')
    parser.add_argument('--regression', action='store_true', help='Check for regression')
    parser.add_argument('--compare', action='store_true', help='Compare configurations')
    parser.add_argument('--conll', action='store_true', help='Run full CoNLL-2003 benchmark')
    
    args = parser.parse_args()
    
    if not any([args.quick, args.full, args.regression, args.compare, args.conll]):
        # Default to quick benchmark
        args.quick = True
    
    try:
        if args.quick:
            result = quick_benchmark()
            print(f"\nQuick benchmark complete. F1: {result['f1']:.3f}, Speed: {result['speed']:.0f} tok/s")
        
        if args.full:
            results = full_benchmark_suite()
            print(f"\nFull benchmark suite complete.")
        
        if args.regression:
            regression_ok = regression_check()
            if regression_ok is False:
                sys.exit(1)  # Exit with error on regression
        
        if args.compare:
            results = compare_configurations()
            print(f"\nConfiguration comparison complete.")
        
        if args.conll:
            try:
                from tests.benchmarks.full_conll_benchmark import run_full_conll_benchmark
                print("Running CoNLL-2003 benchmark (100 samples)...")
                results = run_full_conll_benchmark(max_examples=100)
                print(f"\nCoNLL-2003 benchmark complete. F1-score: {list(results.values())[0].f1:.3f}")
            except ImportError as e:
                print(f"CoNLL-2003 benchmark not available: {e}")
    
    except Exception as e:
        print(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()