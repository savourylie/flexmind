"""
Pytest integration for entity extractor benchmarks.

Run with:
  uv run pytest tests/benchmarks/test_entity_benchmarks.py -v --benchmark-only
  uv run pytest tests/benchmarks/test_entity_benchmarks.py::test_baseline_benchmark -v
"""

import pytest
import time
from pathlib import Path
from flexmind.core.extractors.entities import EntityExtractor
from .utils import EntityBenchmarker
from .benchmark_entity_extractor import create_sample_benchmark_data, create_dialog_benchmark_data


class TestEntityExtractorBenchmarks:
    """Pytest-integrated benchmarks for EntityExtractor."""
    
    @pytest.fixture
    def benchmarker(self):
        """Create benchmarker with default extractor."""
        return EntityBenchmarker()
    
    @pytest.fixture  
    def sample_data(self):
        """Sample benchmark dataset."""
        return create_sample_benchmark_data()
    
    @pytest.fixture
    def dialog_data(self):
        """Dialog-specific benchmark dataset."""
        return create_dialog_benchmark_data()
    
    @pytest.mark.benchmark
    def test_baseline_benchmark(self, benchmarker, sample_data):
        """Establish baseline performance metrics."""
        result = benchmarker.benchmark_on_dataset(sample_data)
        benchmarker.print_results(result)
        
        # Assert minimum performance thresholds
        assert result.f1 >= 0.7, f"F1-score too low: {result.f1:.3f}"
        assert result.precision >= 0.7, f"Precision too low: {result.precision:.3f}"
        assert result.recall >= 0.6, f"Recall too low: {result.recall:.3f}"
        assert result.speed_tokens_per_sec > 1000, f"Speed too slow: {result.speed_tokens_per_sec:.0f} tokens/sec"
        
        # Save as baseline
        baseline_path = Path("tests/benchmarks/baselines/entity_extractor_baseline.json")
        benchmarker.save_results(result, baseline_path)
    
    @pytest.mark.benchmark 
    def test_dialog_benchmark(self, benchmarker, dialog_data):
        """Test performance on dialog-specific data."""
        result = benchmarker.benchmark_on_dataset(dialog_data)
        benchmarker.print_results(result)
        
        # Dialog should have reasonable performance despite challenges
        assert result.f1 >= 0.6, f"Dialog F1-score too low: {result.f1:.3f}"
        assert result.speed_tokens_per_sec > 1000, f"Speed too slow: {result.speed_tokens_per_sec:.0f} tokens/sec"
    
    @pytest.mark.benchmark
    def test_spacy_only_vs_hybrid(self, sample_data):
        """Compare spaCy-only vs hybrid extraction.""" 
        spacy_only = EntityBenchmarker(EntityExtractor(use_fallback=False))
        hybrid = EntityBenchmarker(EntityExtractor(use_fallback=True))
        
        spacy_result = spacy_only.benchmark_on_dataset(sample_data)
        hybrid_result = hybrid.benchmark_on_dataset(sample_data)
        
        print("\nspaCy-only vs Hybrid Comparison:")
        print(f"spaCy-only  | F1: {spacy_result.f1:.3f} | Speed: {spacy_result.speed_tokens_per_sec:.0f} tok/s")
        print(f"Hybrid      | F1: {hybrid_result.f1:.3f} | Speed: {hybrid_result.speed_tokens_per_sec:.0f} tok/s")
        print(f"Fallback rate: {hybrid_result.fallback_rate:.1%}")
        
        # Hybrid might be slower but should have better or equal coverage
        assert hybrid_result.speed_tokens_per_sec > 500, "Hybrid too slow"
        
        # Save comparison results
        results_dir = Path("tests/benchmarks/results")
        spacy_only.save_results(spacy_result, results_dir / "spacy_only.json")
        hybrid.save_results(hybrid_result, results_dir / "hybrid.json")
    
    @pytest.mark.benchmark
    def test_confidence_threshold_impact(self, sample_data):
        """Test how confidence thresholds affect performance."""
        thresholds = [0.5, 0.75, 0.9]
        results = {}
        
        for threshold in thresholds:
            extractor = EntityExtractor(confidence_threshold=threshold)
            benchmarker = EntityBenchmarker(extractor)
            result = benchmarker.benchmark_on_dataset(sample_data)
            results[threshold] = result
        
        print("\nConfidence Threshold Impact:")
        for threshold, result in results.items():
            print(f"Threshold {threshold}: F1={result.f1:.3f}, Precision={result.precision:.3f}, Recall={result.recall:.3f}")
        
        # Higher threshold should generally increase precision, decrease recall
        assert results[0.9].precision >= results[0.5].precision * 0.9, "Higher threshold should maintain precision"
    
    def test_regression_check(self, benchmarker, sample_data):
        """Check for performance regression against baseline."""
        baseline_path = Path("tests/benchmarks/baselines/entity_extractor_baseline.json")
        
        current_result = benchmarker.benchmark_on_dataset(sample_data)
        
        if baseline_path.exists():
            comparison = benchmarker.compare_with_baseline(current_result, baseline_path)
            
            # Assert no significant regression (allow 5% degradation)
            assert comparison.get('f1_change', 0) >= -0.05, f"F1-score regressed significantly: {comparison.get('f1_change', 0):.3f}"
            assert comparison.get('speed_change', 0) >= -1000, f"Speed regressed significantly: {comparison.get('speed_change', 0):.0f} tokens/sec"
        else:
            print("No baseline found - establishing new baseline")
            benchmarker.save_results(current_result, baseline_path)


@pytest.mark.benchmark  
def test_end_to_end_pipeline_benchmark():
    """Benchmark the full preprocessing + extraction pipeline."""
    from flexmind.core.preprocessing.text import TextPreprocessor
    
    preprocessor = TextPreprocessor()
    extractor = EntityExtractor()
    
    # Long document that will be chunked
    long_text = """
    Alice Johnson, a senior researcher at OpenAI in San Francisco, announced a breakthrough in AI alignment.
    The team, led by Dr. Sarah Chen and including Bob Smith from Stanford University, published their findings in Nature.
    The research was funded by a $10 million grant from the National Science Foundation in March 2024.
    President Biden commented on the work during his visit to Silicon Valley last week.
    Tesla CEO Elon Musk also praised the research on Twitter, calling it a significant step forward.
    """
    
    # Test preprocessing + extraction pipeline
    start_time = time.time()
    chunks = preprocessor.process(long_text, text_type='document')
    
    all_entities = []
    for chunk in chunks:
        entities = extractor.extract(chunk.text)
        all_entities.extend(entities)
    
    pipeline_time = time.time() - start_time
    
    print(f"\nEnd-to-End Pipeline Benchmark:")
    print(f"  Input length: {len(long_text)} chars, {len(long_text.split())} tokens")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Entities found: {len(all_entities)}")
    print(f"  Total time: {pipeline_time:.3f} seconds")
    print(f"  Speed: {len(long_text.split())/pipeline_time:.0f} tokens/sec")
    
    # Print found entities
    print(f"  Entities: {[str(e) for e in all_entities[:10]]}")  # Show first 10
    
    # Assert reasonable performance
    assert len(all_entities) >= 5, "Should find multiple entities in long text"
    assert pipeline_time < 5.0, "Pipeline should complete within 5 seconds"
    assert len(long_text.split())/pipeline_time > 50, "Pipeline should process >50 tokens/sec"


if __name__ == "__main__":
    # Run benchmarks directly    
    benchmarker = EntityBenchmarker()
    sample_data = create_sample_benchmark_data()
    
    print("Running Entity Extractor Benchmark Suite...")
    result = benchmarker.benchmark_on_dataset(sample_data)
    benchmarker.print_results(result)