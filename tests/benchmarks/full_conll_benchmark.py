"""
Full CoNLL-2003 benchmarking for EntityExtractor.

Now using the complete 3,453 test sentences for rigorous evaluation
instead of the 5-example mini dataset.
"""

import time
from pathlib import Path
from flexmind.core.extractors.entities import EntityExtractor
from .utils import EntityBenchmarker, load_conll2003_local, get_dataset_stats


def run_full_conll_benchmark(max_examples: int = None, use_fallback: bool = True) -> dict:
    """
    Run comprehensive benchmark on full CoNLL-2003 dataset.
    
    Args:
        max_examples: Limit test examples (None for all 3,453)
        use_fallback: Whether to use DistilBERT fallback
        
    Returns:
        Detailed benchmark results
    """
    print("=" * 70)
    print("FULL CoNLL-2003 BENCHMARK")
    print("=" * 70)
    
    # Load the full test dataset
    print("Loading CoNLL-2003 test dataset...")
    test_data = load_conll2003_local("test", max_examples=max_examples)
    
    print(f"Dataset size: {len(test_data):,} sentences")
    
    # Run benchmark with different configurations
    configurations = {
        'FlexMind Default': EntityExtractor(use_fallback=use_fallback),
        'SpaCy Only': EntityExtractor(use_fallback=False),
        'High Confidence': EntityExtractor(use_fallback=use_fallback, confidence_threshold=0.9),
    }
    
    results = {}
    
    for config_name, extractor in configurations.items():
        print(f"\n{'-' * 50}")
        print(f"TESTING: {config_name}")
        print(f"{'-' * 50}")
        
        benchmarker = EntityBenchmarker(extractor)
        start_time = time.time()
        
        result = benchmarker.benchmark_on_dataset(test_data)
        
        benchmark_time = time.time() - start_time
        result.total_benchmark_time = benchmark_time
        
        benchmarker.print_results(result)
        
        # Save detailed results
        results_dir = Path("benchmark_results/full_conll2003")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filename = config_name.lower().replace(' ', '_') + '.json'
        benchmarker.save_results(result, results_dir / filename)
        
        results[config_name] = result
    
    # Summary comparison
    print(f"\n{'=' * 70}")
    print("CONFIGURATION COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    
    print(f"{'Configuration':<20} {'F1':<6} {'Prec':<6} {'Rec':<6} {'Speed':<10} {'Time':<8}")
    print("-" * 70)
    
    for config_name, result in results.items():
        print(f"{config_name:<20} {result.f1:<6.3f} {result.precision:<6.3f} {result.recall:<6.3f} {result.speed_tokens_per_sec:<10.0f} {result.total_benchmark_time:<8.1f}s")
    
    # Find best performers
    best_f1 = max(results.items(), key=lambda x: x[1].f1)
    best_speed = max(results.items(), key=lambda x: x[1].speed_tokens_per_sec)
    
    print(f"\nBest F1-score: {best_f1[0]} ({best_f1[1].f1:.3f})")
    print(f"Fastest config: {best_speed[0]} ({best_speed[1].speed_tokens_per_sec:.0f} tokens/sec)")
    
    return results


def compare_with_sota():
    """Compare our results with state-of-the-art CoNLL-2003 results."""
    print("\nSOTA Comparison (CoNLL-2003 Test Set):")
    print("-" * 50)
    
    # Known SOTA results from literature
    sota_results = {
        'BERT-Base (Devlin et al.)': {'f1': 0.928},
        'BiLSTM-CRF (Peters et al.)': {'f1': 0.918},
        'Flair (Akbik et al.)': {'f1': 0.936},
        'SpaCy en_core_web_lg': {'f1': 0.85},  # Approximate
    }
    
    # Run our best configuration
    our_extractor = EntityExtractor(use_fallback=True)
    benchmarker = EntityBenchmarker(our_extractor)
    
    # Use subset for quick comparison
    test_data = load_conll2003_local("test", max_examples=500)
    our_result = benchmarker.benchmark_on_dataset(test_data)
    
    print(f"{'Model':<30} {'F1-Score':<10} {'Notes':<20}")
    print("-" * 65)
    
    for model_name, metrics in sota_results.items():
        print(f"{model_name:<30} {metrics['f1']:<10.3f}")
    
    print("-" * 65)
    print(f"{'FlexMind (ours)':<30} {our_result.f1:<10.3f} {'500 test samples':<20}")
    
    # Analysis
    best_sota = max(sota_results.values(), key=lambda x: x['f1'])['f1']
    our_score = our_result.f1
    
    gap = best_sota - our_score
    print(f"\nGap to SOTA: {gap:.3f} ({gap/best_sota*100:.1f}% difference)")
    
    if our_score > 0.8:
        print("✅ Strong performance - suitable for production use")
    elif our_score > 0.7:
        print("⚠️  Good performance - room for improvement")
    else:
        print("❌ Performance needs significant improvement")


def entity_type_analysis():
    """Detailed analysis of performance by entity type."""
    print("\n" + "=" * 70)
    print("ENTITY TYPE ANALYSIS")
    print("=" * 70)
    
    # Get dataset statistics
    stats = get_dataset_stats()
    test_stats = stats['test']
    
    print("CoNLL-2003 Test Set Entity Distribution:")
    total_entities = test_stats['total_entities']
    
    for entity_type, count in sorted(test_stats['entity_type_counts'].items()):
        percentage = count / total_entities * 100
        print(f"  {entity_type:<6}: {count:>4,} entities ({percentage:>5.1f}%)")
    
    # Test on subset to see per-entity performance
    test_data = load_conll2003_local("test", max_examples=1000)
    
    extractor = EntityExtractor(use_fallback=True)
    benchmarker = EntityBenchmarker(extractor)
    result = benchmarker.benchmark_on_dataset(test_data)
    
    print(f"\nFlexMind Performance by Entity Type (1000 test samples):")
    print(f"{'Type':<6} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 50)
    
    for entity_type in ['PER', 'LOC', 'ORG', 'MISC']:
        if entity_type in result.per_entity_metrics:
            metrics = result.per_entity_metrics[entity_type]
            print(f"{entity_type:<6} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1']:<10.3f} {metrics['support']:<10.0f}")
        else:
            print(f"{entity_type:<6} {'No data':<10} {'No data':<10} {'No data':<10} {'0':<10}")


if __name__ == "__main__":
    # Run full benchmark suite
    print("Starting Full CoNLL-2003 Benchmark Suite...")
    
    # First run with subset for quick results
    print("Quick validation run (100 samples):")
    quick_results = run_full_conll_benchmark(max_examples=100)
    
    print("\n" + "="*70)
    response = input("Run full benchmark on all 3,453 test samples? (y/n): ")
    
    if response.lower().startswith('y'):
        print("Running full benchmark - this may take several minutes...")
        full_results = run_full_conll_benchmark()
        
        # Additional analyses
        entity_type_analysis()
        compare_with_sota()
        
        print(f"\n✅ Full benchmark complete!")
        print(f"Results saved to: benchmark_results/full_conll2003/")
    else:
        print("Skipping full benchmark. Use quick results for development.")
    
    print("\nTo run specific tests:")
    print("  uv run python tests/benchmarks/full_conll_benchmark.py")
    print("  uv run pytest tests/benchmarks/test_entity_benchmarks.py::test_baseline_benchmark -v")