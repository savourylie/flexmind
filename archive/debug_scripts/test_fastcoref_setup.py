#!/usr/bin/env python3
"""Test fastcoref installation and basic functionality."""

import fastcoref
from fastcoref import FCoref

def test_basic_functionality():
    """Test that fastcoref works with a simple example."""
    
    print("=== Testing Fastcoref Installation ===")
    try:
        print(f"Fastcoref version: {fastcoref.__version__}")
    except AttributeError:
        print("Fastcoref module loaded successfully (version info not available)")
    
    # Test basic coreference resolution
    text = """Alice went to the store. She bought some milk. Alice also purchased bread because she needed it for dinner."""
    
    print(f"\nInput text: {text}")
    
    # Initialize model
    print("\nInitializing FCoref model...")
    try:
        model = FCoref(device='cpu')  # Use CPU to avoid GPU requirements
    except Exception as e:
        print(f"Error initializing FCoref: {e}")
        print("Trying default initialization...")
        model = FCoref()
    
    # Run coreference resolution
    print("Running coreference resolution...")
    predictions = model.predict(texts=[text])
    
    print(f"\nPredictions: {predictions}")
    
    # Print clusters if available
    if predictions and len(predictions) > 0 and predictions[0]:
        print("\nCoreference clusters:")
        for i, cluster in enumerate(predictions[0]):
            print(f"  Cluster {i+1}: {cluster}")
    
    print("\n=== Fastcoref test completed successfully! ===")

if __name__ == "__main__":
    test_basic_functionality()