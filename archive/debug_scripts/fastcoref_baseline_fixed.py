#!/usr/bin/env python3
"""Fixed fastcoref baseline evaluation with correct output format handling."""

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
            "expected_challenge": "Multiple same-gender entities with ambiguous pronoun resolution"
        },
        {
            "id": "john_mary_project", 
            "text": "John works with Mary on the project. He thinks her approach is innovative, but she believes his methodology needs improvement.",
            "description": "Different-gender entities with clear pronouns",
            "difficulty": "easy",
            "expected_challenge": "Should be handled well by existing systems"
        },
        {
            "id": "three_way_ambiguity",
            "text": "Sarah, Lisa, and Emma went to the conference. She presented first, then she asked her a question, and finally she responded.",
            "description": "Three-way ambiguity with multiple pronouns",
            "difficulty": "very_high", 
            "expected_challenge": "Highly ambiguous - coref systems typically struggle"
        },
        {
            "id": "organization_person",
            "text": "Microsoft hired Sarah as a director. The company believes she will transform their cloud strategy.",
            "description": "Organization vs person entity",
            "difficulty": "medium",
            "expected_challenge": "Should distinguish organization (their) from person (she)"
        },
        {
            "id": "long_distance_reference",
            "text": "Dr. Johnson published a groundbreaking study last year. The research examined climate patterns across multiple decades. Many scientists praised the methodology. However, some critics argued that the conclusions were overstated. Despite the controversy, she continues to defend her work.",
            "description": "Long-distance pronoun reference",
            "difficulty": "medium",
            "expected_challenge": "Long distance between 'Dr. Johnson' and 'she/her'"
        }
    ]
    
    return test_cases

def extract_cluster_text(coref_result, cluster_positions) -> List[str]:
    """Extract text for a cluster from CorefResult object."""
    
    cluster_texts = []
    text = coref_result.text
    
    # Use char_map to convert token positions to character positions
    for token_start, token_end in cluster_positions:
        # Look up character positions in char_map
        if (token_start, token_end) in coref_result.char_map:
            _, (char_start, char_end) = coref_result.char_map[(token_start, token_end)]
            span_text = text[char_start:char_end]
            cluster_texts.append(span_text)
    
    return cluster_texts

def analyze_predictions(case: Dict[str, Any], predictions: List[Any]) -> Dict[str, Any]:
    """Analyze fastcoref predictions for a test case."""
    
    analysis = {
        "case_id": case["id"],
        "text": case["text"],
        "difficulty": case["difficulty"],
        "raw_predictions": str(predictions),  # Keep raw for debugging
        "analysis": {}
    }
    
    if predictions and len(predictions) > 0 and hasattr(predictions[0], 'clusters'):
        coref_result = predictions[0]
        raw_clusters = coref_result.clusters
        
        # Extract text for each cluster
        text_clusters = []
        for cluster_positions in raw_clusters:
            cluster_text = extract_cluster_text(coref_result, cluster_positions)
            if cluster_text:  # Only add non-empty clusters
                text_clusters.append(cluster_text)
        
        analysis["analysis"] = {
            "num_clusters": len(text_clusters),
            "clusters": text_clusters,
            "cluster_sizes": [len(cluster) for cluster in text_clusters],
            "total_entities": sum(len(cluster) for cluster in text_clusters)
        }
        
        # Analyze pronoun patterns
        pronoun_patterns = {"she", "her", "he", "him", "his", "they", "them", "their"}
        
        pronoun_clusters = []
        for cluster in text_clusters:
            cluster_lower = [span.lower() for span in cluster]
            if any(pronoun in cluster_lower for pronoun in pronoun_patterns):
                pronoun_clusters.append(cluster)
        
        analysis["analysis"]["pronoun_clusters"] = pronoun_clusters
        analysis["analysis"]["has_pronoun_resolution"] = len(pronoun_clusters) > 0
        
        # Check for our specific challenging patterns
        if case["id"] == "alice_beth_cafe":
            # Check if both Alice and Beth are assigned pronouns
            alice_cluster = None
            beth_cluster = None
            for cluster in text_clusters:
                cluster_lower = [span.lower() for span in cluster]
                if "alice" in cluster_lower:
                    alice_cluster = cluster
                if "beth" in cluster_lower:
                    beth_cluster = cluster
            
            analysis["analysis"]["alice_cluster"] = alice_cluster
            analysis["analysis"]["beth_cluster"] = beth_cluster
            analysis["analysis"]["ambiguity_handled"] = alice_cluster is not None and beth_cluster is not None
        
    else:
        analysis["analysis"] = {
            "num_clusters": 0,
            "clusters": [],
            "error": "No valid predictions returned"
        }
    
    return analysis

def evaluate_fastcoref_baseline() -> Dict[str, Any]:
    """Run comprehensive baseline evaluation of fastcoref."""
    
    print("=== Fixed Fastcoref Baseline Evaluation ===")
    
    # Create test cases
    test_cases = create_challenging_test_cases()
    print(f"Created {len(test_cases)} challenging test cases")
    
    # Test FCoref model (faster one)
    model_name = "FCoref"
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize model
        print("Initializing model...")
        start_init = time.time()
        model = FCoref()
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
                
                # Special analysis for Alice/Beth case
                if case["id"] == "alice_beth_cafe":
                    alice_c = analysis['analysis'].get('alice_cluster')
                    beth_c = analysis['analysis'].get('beth_cluster')
                    print(f"  Alice cluster: {alice_c}")
                    print(f"  Beth cluster: {beth_c}")
                    print(f"  Ambiguity handled: {analysis['analysis'].get('ambiguity_handled')}")
                
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
        if model_results['summary']['avg_time_per_case']:
            print(f"  Average time per case: {model_results['summary']['avg_time_per_case']:.3f}s")
        print(f"  Success rate: {model_results['summary']['success_rate']:.2%}")
        
        return model_results
        
    except Exception as e:
        print(f"Failed to evaluate {model_name}: {e}")
        return {
            "error": str(e),
            "model_name": model_name
        }

def save_results(results: Dict[str, Any], filename: str = "fastcoref_baseline_results_fixed.json"):
    """Save evaluation results to file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)  # default=str for numpy types
    print(f"\nResults saved to {filename}")

def analyze_alice_beth_performance(results: Dict[str, Any]):
    """Analyze performance specifically on the Alice/Beth café scenario."""
    
    print(f"\n{'='*60}")
    print("ALICE/BETH CAFÉ SCENARIO ANALYSIS")
    print(f"{'='*60}")
    
    if "error" in results:
        print("Evaluation failed - cannot analyze Alice/Beth performance")
        return
    
    # Find the Alice/Beth test case results
    alice_beth_result = None
    for test_result in results["test_results"]:
        if test_result.get("case_id") == "alice_beth_cafe":
            alice_beth_result = test_result
            break
    
    if not alice_beth_result:
        print("Alice/Beth test case not found")
        return
    
    print(f"Text: {alice_beth_result['text']}")
    print(f"Processing time: {alice_beth_result.get('processing_time', 'N/A')}s")
    
    analysis = alice_beth_result['analysis']
    print(f"\nClusters found: {analysis.get('num_clusters', 0)}")
    
    clusters = analysis.get('clusters', [])
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {cluster}")
    
    alice_cluster = analysis.get('alice_cluster')
    beth_cluster = analysis.get('beth_cluster') 
    ambiguity_handled = analysis.get('ambiguity_handled', False)
    
    print(f"\nDetailed Analysis:")
    print(f"  Alice cluster: {alice_cluster}")
    print(f"  Beth cluster: {beth_cluster}")
    print(f"  Ambiguity resolution: {'SUCCESS' if ambiguity_handled else 'PARTIAL/FAILED'}")
    
    # This is exactly the type of case our late chunking method should improve!
    if not ambiguity_handled:
        print(f"\n🎯 RESEARCH OPPORTUNITY:")
        print(f"  This case shows current limitations of fastcoref")
        print(f"  Our late chunking method should improve disambiguation here")

if __name__ == "__main__":
    results = evaluate_fastcoref_baseline()
    save_results(results)
    analyze_alice_beth_performance(results)