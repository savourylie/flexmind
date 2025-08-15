#!/usr/bin/env python3
"""Fastcoref baseline evaluation with challenging test cases similar to OntoNotes scenarios."""

from fastcoref import FCoref, LingMessCoref
import time
from typing import List, Dict, Any
import json

def create_challenging_test_cases() -> List[Dict[str, Any]]:
    """Create test cases that represent the challenging scenarios we want to solve."""
    
    test_cases = [
        # Multi-entity ambiguous pronouns (our core research target)
        {
            "id": "alice_beth_cafe",
            "text": "When Alice met Beth at the café, she told her that she had won the lottery.",
            "description": "Alice/Beth café scenario - core research target",
            "difficulty": "high",
            "expected_challenge": "Multiple same-gender entities with ambiguous pronoun resolution",
            "gold_clusters": None  # Ambiguous - multiple valid interpretations
        },
        {
            "id": "john_mary_project", 
            "text": "John works with Mary on the project. He thinks her approach is innovative, but she believes his methodology needs improvement.",
            "description": "Different-gender entities with clear pronouns",
            "difficulty": "easy",
            "expected_challenge": "Should be handled well by existing systems",
            "gold_clusters": [
                ["John", "He", "his"],
                ["Mary", "her", "she"] 
            ]
        },
        {
            "id": "three_way_ambiguity",
            "text": "Sarah, Lisa, and Emma went to the conference. She presented first, then she asked her a question, and finally she responded.",
            "description": "Three-way ambiguity with multiple pronouns",
            "difficulty": "very_high", 
            "expected_challenge": "Highly ambiguous - coref systems typically struggle",
            "gold_clusters": None  # Highly ambiguous
        },
        {
            "id": "organization_person",
            "text": "Microsoft hired Sarah as a director. The company believes she will transform their cloud strategy.",
            "description": "Organization vs person entity",
            "difficulty": "medium",
            "expected_challenge": "Should distinguish organization (their) from person (she)",
            "gold_clusters": [
                ["Microsoft", "company", "their"],
                ["Sarah", "she"]
            ]
        },
        {
            "id": "long_distance_reference",
            "text": "Dr. Johnson published a groundbreaking study last year. The research examined climate patterns across multiple decades. Many scientists praised the methodology. However, some critics argued that the conclusions were overstated. Despite the controversy, she continues to defend her work.",
            "description": "Long-distance pronoun reference",
            "difficulty": "medium",
            "expected_challenge": "Long distance between 'Dr. Johnson' and 'she/her'",
            "gold_clusters": [
                ["Dr. Johnson", "she", "her"],
                ["study", "research", "methodology", "conclusions", "work"]
            ]
        },
        {
            "id": "same_profession_ambiguity",
            "text": "The lawyer met with the judge before the trial. He was concerned about the evidence, and he wanted to discuss procedural matters.",
            "description": "Same-profession same-gender ambiguity", 
            "difficulty": "high",
            "expected_challenge": "Professional context with gender ambiguity",
            "gold_clusters": None  # Ambiguous without more context
        },
        {
            "id": "nested_entities",
            "text": "The CEO of the startup, Maria Rodriguez, announced the acquisition. She said that the company she founded would maintain its independence under the new structure.",
            "description": "Nested entities with multiple 'she' references",
            "difficulty": "medium-high",
            "expected_challenge": "Nested company/person entities with pronoun disambiguation", 
            "gold_clusters": [
                ["CEO", "Maria Rodriguez", "She", "she"],
                ["startup", "company"],
                ["acquisition", "new structure"]
            ]
        }
    ]
    
    return test_cases

def analyze_predictions(case: Dict[str, Any], predictions: List[Any]) -> Dict[str, Any]:
    """Analyze fastcoref predictions for a test case."""
    
    analysis = {
        "case_id": case["id"],
        "text": case["text"],
        "difficulty": case["difficulty"],
        "predictions": predictions,
        "analysis": {}
    }
    
    if predictions and len(predictions) > 0 and predictions[0] and hasattr(predictions[0], 'clusters'):
        clusters = predictions[0].clusters
        
        analysis["analysis"] = {
            "num_clusters": len(clusters),
            "clusters": clusters,
            "cluster_sizes": [len(cluster) for cluster in clusters],
            "total_entities": sum(len(cluster) for cluster in clusters)
        }
        
        # Check for specific patterns we care about
        pronoun_patterns = {"she", "her", "he", "him", "his", "they", "them", "their"}
        
        pronoun_clusters = []
        for cluster in clusters:
            cluster_lower = [span.lower() for span in cluster]
            if any(pronoun in cluster_lower for pronoun in pronoun_patterns):
                pronoun_clusters.append(cluster)
        
        analysis["analysis"]["pronoun_clusters"] = pronoun_clusters
        analysis["analysis"]["has_pronoun_resolution"] = len(pronoun_clusters) > 0
        
    else:
        analysis["analysis"] = {
            "num_clusters": 0,
            "clusters": [],
            "error": "No valid predictions returned"
        }
    
    return analysis

def evaluate_fastcoref_baseline() -> Dict[str, Any]:
    """Run comprehensive baseline evaluation of fastcoref."""
    
    print("=== Fastcoref Baseline Evaluation ===")
    
    # Create test cases
    test_cases = create_challenging_test_cases()
    print(f"Created {len(test_cases)} challenging test cases")
    
    # Test both models
    models_to_test = [
        ("FCoref", FCoref),
        ("LingMessCoref", LingMessCoref)
    ]
    
    results = {}
    
    for model_name, model_class in models_to_test:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        try:
            # Initialize model
            print("Initializing model...")
            start_init = time.time()
            model = model_class()
            init_time = time.time() - start_init
            print(f"Model initialized in {init_time:.2f}s")
            
            model_results = {
                "model_name": model_name,
                "init_time": init_time,
                "test_results": [],
                "summary": {}
            }
            
            total_time = 0
            successful_cases = 0
            
            # Run evaluation on each test case
            for case in test_cases:
                print(f"\n--- {case['id']}: {case['description']} ---")
                print(f"Difficulty: {case['difficulty']}")
                print(f"Text: {case['text']}")
                
                try:
                    start_time = time.time()
                    predictions = model.predict([case['text']])
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    total_time += processing_time
                    
                    print(f"Processing time: {processing_time:.3f}s")
                    
                    # Analyze predictions
                    analysis = analyze_predictions(case, predictions)
                    analysis["processing_time"] = processing_time
                    
                    print(f"Clusters found: {analysis['analysis'].get('num_clusters', 0)}")
                    if analysis['analysis'].get('clusters'):
                        for i, cluster in enumerate(analysis['analysis']['clusters']):
                            print(f"  Cluster {i+1}: {cluster}")
                    
                    model_results["test_results"].append(analysis)
                    successful_cases += 1
                    
                except Exception as e:
                    print(f"Error processing case: {e}")
                    model_results["test_results"].append({
                        "case_id": case["id"],
                        "error": str(e),
                        "processing_time": None
                    })
            
            # Create summary
            model_results["summary"] = {
                "total_cases": len(test_cases),
                "successful_cases": successful_cases,
                "total_time": total_time,
                "avg_time_per_case": total_time / successful_cases if successful_cases > 0 else None,
                "success_rate": successful_cases / len(test_cases)
            }
            
            print(f"\n{model_name} Summary:")
            print(f"  Successful cases: {successful_cases}/{len(test_cases)}")
            print(f"  Average time per case: {model_results['summary']['avg_time_per_case']:.3f}s")
            print(f"  Success rate: {model_results['summary']['success_rate']:.2%}")
            
            results[model_name] = model_results
            
        except Exception as e:
            print(f"Failed to evaluate {model_name}: {e}")
            results[model_name] = {
                "error": str(e),
                "model_name": model_name
            }
    
    return results

def save_results(results: Dict[str, Any], filename: str = "fastcoref_baseline_results.json"):
    """Save evaluation results to file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

def print_comparative_summary(results: Dict[str, Any]):
    """Print comparative summary of model performance."""
    
    print(f"\n{'='*60}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*60}")
    
    for model_name, model_results in results.items():
        if "error" not in model_results:
            summary = model_results["summary"]
            print(f"\n{model_name}:")
            print(f"  Success Rate: {summary['success_rate']:.2%}")
            print(f"  Avg Time/Case: {summary['avg_time_per_case']:.3f}s")
            print(f"  Total Time: {summary['total_time']:.2f}s")
        else:
            print(f"\n{model_name}: FAILED - {model_results['error']}")

if __name__ == "__main__":
    results = evaluate_fastcoref_baseline()
    save_results(results)
    print_comparative_summary(results)