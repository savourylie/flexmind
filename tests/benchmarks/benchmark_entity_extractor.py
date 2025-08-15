"""
Comprehensive benchmarking suite for EntityExtractor.

Tests against standard NER datasets and domain-specific examples to establish
performance baselines and detect regressions.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

import pytest
from flexmind.core.extractors.entities import EntityExtractor


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    precision: float
    recall: float
    f1: float
    accuracy: float
    speed_tokens_per_sec: float
    total_time: float
    total_tokens: int
    total_entities_predicted: int
    total_entities_actual: int
    fallback_rate: float
    per_entity_metrics: Dict[str, Dict[str, float]]


@dataclass
class AnnotatedExample:
    """A text example with ground truth entity annotations."""
    text: str
    entities: List[Tuple[str, str, int, int]]  # (text, label, start, end)
    metadata: Dict[str, Any] = None


class EntityBenchmarker:
    """
    Comprehensive benchmarking for EntityExtractor performance.
    
    Features:
    - Standard dataset evaluation (CoNLL-2003 format)  
    - Domain-specific benchmarks (dialog, modern entities)
    - Performance regression testing
    - Speed benchmarking
    - Per-entity-type analysis
    """
    
    def __init__(self, extractor: EntityExtractor = None):
        """Initialize with optional custom extractor."""
        self.extractor = extractor or EntityExtractor()
        self.results_history = []
    
    def benchmark_on_dataset(self, examples: List[AnnotatedExample]) -> BenchmarkResult:
        """
        Run comprehensive benchmark on a dataset.
        
        Args:
            examples: List of annotated examples to evaluate on
            
        Returns:
            BenchmarkResult with detailed metrics
        """
        print(f"Running benchmark on {len(examples)} examples...")
        
        all_predictions = []
        all_ground_truth = []
        total_time = 0
        total_tokens = 0
        fallback_count = 0
        per_entity_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for i, example in enumerate(examples):
            if i % 50 == 0:
                print(f"  Progress: {i+1}/{len(examples)} examples")
            
            # Time the extraction
            start_time = time.time()
            predicted_entities = self.extractor.extract(example.text)
            extraction_time = time.time() - start_time
            
            total_time += extraction_time
            total_tokens += len(example.text.split())
            
            # Count fallback usage (simplified heuristic)
            fallback_count += sum(1 for e in predicted_entities if e.source == 'hybrid')
            
            # Convert predictions to comparable format
            pred_set = set()
            for entity in predicted_entities:
                # For benchmarking, we match by text and label
                pred_set.add((entity.text.lower().strip(), entity.label))
            
            # Convert ground truth to comparable format  
            gt_set = set()
            for entity_text, label, start, end in example.entities:
                gt_set.add((entity_text.lower().strip(), self._normalize_label(label)))
            
            all_predictions.append(pred_set)
            all_ground_truth.append(gt_set)
            
            # Calculate per-entity-type metrics
            self._update_per_entity_stats(pred_set, gt_set, per_entity_stats)
        
        # Calculate overall metrics
        precision, recall, f1, accuracy = self._calculate_metrics(all_predictions, all_ground_truth)
        
        # Calculate per-entity-type metrics
        per_entity_metrics = {}
        for entity_type, stats in per_entity_stats.items():
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            if tp + fp > 0:
                prec = tp / (tp + fp)
            else:
                prec = 0.0
            if tp + fn > 0:
                rec = tp / (tp + fn)  
            else:
                rec = 0.0
            if prec + rec > 0:
                f1_score = 2 * prec * rec / (prec + rec)
            else:
                f1_score = 0.0
                
            per_entity_metrics[entity_type] = {
                'precision': prec,
                'recall': rec, 
                'f1': f1_score,
                'support': tp + fn
            }
        
        result = BenchmarkResult(
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            speed_tokens_per_sec=total_tokens / total_time if total_time > 0 else 0,
            total_time=total_time,
            total_tokens=total_tokens,
            total_entities_predicted=sum(len(pred) for pred in all_predictions),
            total_entities_actual=sum(len(gt) for gt in all_ground_truth),
            fallback_rate=fallback_count / len(examples) if examples else 0,
            per_entity_metrics=per_entity_metrics
        )
        
        self.results_history.append(result)
        return result
    
    def _calculate_metrics(self, predictions: List[set], ground_truth: List[set]) -> Tuple[float, float, float, float]:
        """Calculate precision, recall, F1, and accuracy."""
        total_tp = 0
        total_fp = 0  
        total_fn = 0
        exact_matches = 0
        
        for pred_set, gt_set in zip(predictions, ground_truth):
            tp = len(pred_set & gt_set)  # True positives
            fp = len(pred_set - gt_set)  # False positives
            fn = len(gt_set - pred_set)  # False negatives
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            if pred_set == gt_set:
                exact_matches += 1
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = exact_matches / len(predictions) if predictions else 0
        
        return precision, recall, f1, accuracy
    
    def _update_per_entity_stats(self, pred_set: set, gt_set: set, stats: dict):
        """Update per-entity-type statistics."""
        # True positives
        for pred_text, pred_label in pred_set & gt_set:
            stats[pred_label]['tp'] += 1
        
        # False positives
        for pred_text, pred_label in pred_set - gt_set:
            stats[pred_label]['fp'] += 1
        
        # False negatives
        for gt_text, gt_label in gt_set - pred_set:
            stats[gt_label]['fn'] += 1
    
    def _normalize_label(self, label: str) -> str:
        """Normalize entity labels for comparison."""
        # Map various label formats to standard ones
        label_mapping = {
            'PER': 'PERSON',
            'PERSON': 'PERSON', 
            'LOC': 'GPE',
            'LOCATION': 'GPE',
            'GPE': 'GPE',
            'ORG': 'ORG',
            'ORGANIZATION': 'ORG',
            'MISC': 'MISC',
            'DATE': 'DATE',
            'TIME': 'TIME',
            'MONEY': 'MONEY',
            'PERCENT': 'PERCENT'
        }
        return label_mapping.get(label.upper(), label.upper())
    
    def print_results(self, result: BenchmarkResult):
        """Print formatted benchmark results."""
        print("\n" + "=" * 70)
        print("ENTITY EXTRACTOR BENCHMARK RESULTS")
        print("=" * 70)
        
        print(f"Overall Metrics:")
        print(f"  Precision: {result.precision:.3f}")
        print(f"  Recall:    {result.recall:.3f}")
        print(f"  F1-Score:  {result.f1:.3f}")
        print(f"  Accuracy:  {result.accuracy:.3f}")
        
        print(f"\nPerformance:")
        print(f"  Speed:           {result.speed_tokens_per_sec:.0f} tokens/sec")
        print(f"  Total time:      {result.total_time:.2f} seconds")
        print(f"  Fallback rate:   {result.fallback_rate:.1%}")
        
        print(f"\nEntity Counts:")
        print(f"  Predicted: {result.total_entities_predicted}")
        print(f"  Actual:    {result.total_entities_actual}")
        
        print(f"\nPer-Entity-Type Performance:")
        for entity_type, metrics in result.per_entity_metrics.items():
            support = metrics['support']
            print(f"  {entity_type:8} | P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f} | Support: {support}")
    
    def save_results(self, result: BenchmarkResult, filepath: Path):
        """Save benchmark results to JSON file."""
        result_dict = {
            'precision': result.precision,
            'recall': result.recall,
            'f1': result.f1,
            'accuracy': result.accuracy,
            'speed_tokens_per_sec': result.speed_tokens_per_sec,
            'total_time': result.total_time,
            'total_tokens': result.total_tokens,
            'total_entities_predicted': result.total_entities_predicted,
            'total_entities_actual': result.total_entities_actual,
            'fallback_rate': result.fallback_rate,
            'per_entity_metrics': result.per_entity_metrics,
            'timestamp': time.time()
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def compare_with_baseline(self, current: BenchmarkResult, baseline_path: Path) -> Dict[str, float]:
        """Compare current results with baseline."""
        if not baseline_path.exists():
            print(f"No baseline found at {baseline_path}")
            return {}
        
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        
        comparison = {
            'f1_change': current.f1 - baseline_data['f1'],
            'precision_change': current.precision - baseline_data['precision'],
            'recall_change': current.recall - baseline_data['recall'],
            'speed_change': current.speed_tokens_per_sec - baseline_data['speed_tokens_per_sec'],
        }
        
        print(f"\nComparison with baseline:")
        for metric, change in comparison.items():
            direction = "↑" if change > 0 else "↓" if change < 0 else "="
            print(f"  {metric}: {change:+.3f} {direction}")
        
        return comparison


def create_sample_benchmark_data() -> List[AnnotatedExample]:
    """Create sample annotated data for testing."""
    return [
        AnnotatedExample(
            text="Alice Johnson works at OpenAI in San Francisco.",
            entities=[
                ("Alice Johnson", "PERSON", 0, 13),
                ("OpenAI", "ORG", 23, 29),
                ("San Francisco", "GPE", 33, 46)
            ]
        ),
        AnnotatedExample(
            text="President Biden met with Elon Musk in Washington D.C. on January 15th, 2024.",
            entities=[
                ("Biden", "PERSON", 10, 15),
                ("Elon Musk", "PERSON", 25, 34),
                ("Washington D.C.", "GPE", 38, 52),
                ("January 15th, 2024", "DATE", 56, 74)
            ]
        ),
        AnnotatedExample(
            text="Apple CEO Tim Cook announced a $50 billion investment plan.",
            entities=[
                ("Apple", "ORG", 0, 5),
                ("Tim Cook", "PERSON", 10, 18),
                ("$50 billion", "MONEY", 32, 43)
            ]
        ),
        AnnotatedExample(
            text="The AI researcher Dr. Sarah Chen works at DeepMind in London.",
            entities=[
                ("Dr. Sarah Chen", "PERSON", 18, 32),
                ("DeepMind", "ORG", 42, 50),
                ("London", "GPE", 54, 60)
            ]
        ),
        AnnotatedExample(
            text="Google acquired the startup for €100 million in March 2023.",
            entities=[
                ("Google", "ORG", 0, 6),
                ("€100 million", "MONEY", 32, 44),
                ("March 2023", "DATE", 48, 58)
            ]
        )
    ]


def create_dialog_benchmark_data() -> List[AnnotatedExample]:
    """Create dialog-specific benchmark data.""" 
    return [
        AnnotatedExample(
            text="Alice: I just started working at Meta! Bob: That's great! When did you move to Menlo Park?",
            entities=[
                ("Alice", "PERSON", 0, 5),
                ("Meta", "ORG", 33, 37),
                ("Bob", "PERSON", 39, 42),
                ("Menlo Park", "GPE", 82, 92)
            ],
            metadata={"type": "dialog"}
        ),
        AnnotatedExample(
            text="Sarah: Hey, did you see that Tesla's stock hit $300? Mike: Yeah, Elon tweeted about it yesterday.",
            entities=[
                ("Sarah", "PERSON", 0, 5),
                ("Tesla", "ORG", 28, 33),
                ("$300", "MONEY", 48, 52),
                ("Mike", "PERSON", 54, 58),
                ("Elon", "PERSON", 66, 70)
            ],
            metadata={"type": "dialog"}
        )
    ]


if __name__ == "__main__":
    # Quick benchmark run
    benchmarker = EntityBenchmarker()
    
    print("Running sample benchmark...")
    sample_data = create_sample_benchmark_data() 
    result = benchmarker.benchmark_on_dataset(sample_data)
    benchmarker.print_results(result)
    
    print("\nRunning dialog benchmark...")
    dialog_data = create_dialog_benchmark_data()
    dialog_result = benchmarker.benchmark_on_dataset(dialog_data)
    benchmarker.print_results(dialog_result)
    
    # Save results
    results_dir = Path("benchmark_results")
    benchmarker.save_results(result, results_dir / "sample_benchmark.json")
    benchmarker.save_results(dialog_result, results_dir / "dialog_benchmark.json")