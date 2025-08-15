#!/usr/bin/env python3
"""Debug fastcoref output format to understand the data structure."""

from fastcoref import FCoref

def debug_fastcoref_output():
    """Debug the actual output format from fastcoref."""
    
    print("=== Debugging Fastcoref Output Format ===")
    
    # Simple test text
    text = "John went to the store. He bought some milk."
    
    print(f"Input text: {text}")
    
    # Initialize model
    print("Initializing FCoref...")
    model = FCoref()
    
    # Get predictions
    print("Running prediction...")
    predictions = model.predict([text])
    
    print(f"\nType of predictions: {type(predictions)}")
    print(f"Length of predictions: {len(predictions)}")
    
    if predictions:
        print(f"\nFirst prediction type: {type(predictions[0])}")
        print(f"First prediction: {predictions[0]}")
        
        if hasattr(predictions[0], '__dict__'):
            print(f"Prediction attributes: {predictions[0].__dict__}")
        
        if hasattr(predictions[0], 'clusters'):
            clusters = predictions[0].clusters
            print(f"\nClusters type: {type(clusters)}")
            print(f"Number of clusters: {len(clusters)}")
            
            for i, cluster in enumerate(clusters):
                print(f"\nCluster {i+1}:")
                print(f"  Type: {type(cluster)}")
                print(f"  Content: {cluster}")
                
                if cluster:  # If cluster is not empty
                    print(f"  First element type: {type(cluster[0])}")
                    print(f"  First element: {cluster[0]}")
                    
                    # If it's a tuple or list, examine its structure
                    if isinstance(cluster[0], (tuple, list)):
                        print(f"  First element length: {len(cluster[0])}")
                        for j, item in enumerate(cluster[0]):
                            print(f"    Item {j}: {item} (type: {type(item)})")

def test_with_challenging_case():
    """Test with one of our challenging cases."""
    
    print("\n" + "="*60)
    print("Testing with Alice/Beth case")
    print("="*60)
    
    text = "When Alice met Beth at the café, she told her that she had won the lottery."
    print(f"Input: {text}")
    
    model = FCoref()
    predictions = model.predict([text])
    
    print(f"Predictions: {predictions}")
    
    if predictions and predictions[0] and hasattr(predictions[0], 'clusters'):
        print(f"Number of clusters found: {len(predictions[0].clusters)}")
        for i, cluster in enumerate(predictions[0].clusters):
            print(f"Cluster {i+1}: {cluster}")

if __name__ == "__main__":
    debug_fastcoref_output()
    test_with_challenging_case()