#!/usr/bin/env python3
"""Basic fastcoref evaluation with custom test cases."""

from fastcoref import FCoref, LingMessCoref
import time

def create_test_cases():
    """Create test cases similar to our target scenarios."""
    
    # Cases similar to GAP and multi-entity scenarios
    test_cases = [
        {
            "text": "Alice met Beth at the café. She told her that she had won the lottery.",
            "description": "Multi-entity ambiguous pronouns (Alice/Beth café scenario)",
            "expected_clusters": [
                ["Alice", "She"], ["Beth", "her", "she"]  # One possible interpretation
            ]
        },
        {
            "text": "John works with Mary. He likes her project, but she thinks his approach is better.",
            "description": "Gender-different entities with clear pronoun resolution",
            "expected_clusters": [
                ["John", "He", "his"], ["Mary", "her", "she"]
            ]
        },
        {
            "text": "The company hired Sarah. The organization believes she will improve their efficiency.",
            "description": "Organization vs. person entity resolution",
            "expected_clusters": [
                ["company", "organization", "their"], ["Sarah", "she"]
            ]
        },
        {
            "text": "When the professor met the student, he gave him some feedback about his thesis.",
            "description": "Same-gender ambiguous entities",
            "expected_clusters": []  # Ambiguous - could be interpreted multiple ways
        },
        {
            "text": "Lisa and Emma went shopping. She bought a dress while she chose shoes.",
            "description": "Two same-gender entities with ambiguous pronouns",
            "expected_clusters": []  # Highly ambiguous
        }
    ]
    
    return test_cases

def evaluate_model(model, test_cases):
    """Evaluate model on test cases."""
    
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {case['description']} ---")
        print(f"Text: {case['text']}")
        
        try:
            start_time = time.time()
            predictions = model.predict([case['text']])
            end_time = time.time()
            
            print(f"Predictions: {predictions}")
            print(f"Processing time: {end_time - start_time:.3f}s")
            
            # Analyze the predictions
            if predictions and len(predictions) > 0 and predictions[0]:
                print("Detected clusters:")
                for j, cluster in enumerate(predictions[0]):
                    cluster_text = [case['text'][start:end] for start, end in cluster]
                    print(f"  Cluster {j+1}: {cluster_text}")
            else:
                print("No clusters detected")
            
            results.append({
                "case": case,
                "predictions": predictions,
                "time": end_time - start_time
            })
            
        except Exception as e:
            print(f"Error processing case: {e}")
            results.append({
                "case": case,
                "predictions": None,
                "error": str(e),
                "time": None
            })
    
    return results

def run_evaluation():
    """Run complete evaluation."""
    
    print("=== Fastcoref Baseline Evaluation ===")
    
    # Create test cases
    test_cases = create_test_cases()
    print(f"Created {len(test_cases)} test cases")
    
    # Test different models
    models_to_test = [
        ("FCoref (Fast)", FCoref),
        ("LingMessCoref (Accurate)", LingMessCoref)
    ]
    
    all_results = {}
    
    for model_name, model_class in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing {model_name}")
        print(f"{'='*50}")
        
        try:
            print("Initializing model...")
            model = model_class()
            print(f"{model_name} initialized successfully!")
            
            # Run evaluation
            results = evaluate_model(model, test_cases)
            all_results[model_name] = results
            
        except Exception as e:
            print(f"Failed to initialize {model_name}: {e}")
            all_results[model_name] = None
    
    # Summary
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    
    for model_name, results in all_results.items():
        if results is None:
            print(f"{model_name}: FAILED TO INITIALIZE")
        else:
            successful_cases = len([r for r in results if r['predictions'] is not None])
            avg_time = sum(r['time'] for r in results if r['time'] is not None) / len(results)
            print(f"{model_name}: {successful_cases}/{len(test_cases)} cases successful, avg time: {avg_time:.3f}s")
    
    return all_results

if __name__ == "__main__":
    results = run_evaluation()