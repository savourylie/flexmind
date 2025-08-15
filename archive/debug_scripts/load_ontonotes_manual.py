#!/usr/bin/env python3
"""Manual approach to load OntoNotes data for evaluation."""

import requests
import json

def try_simple_fastcoref_test():
    """Try a simple fastcoref test first."""
    print("=== Simple Fastcoref Test ===")
    
    try:
        from fastcoref import FCoref
        
        # Test with simple text first
        text = "John went to the store. He bought some milk."
        print(f"Test text: {text}")
        
        # Try to initialize with a smaller/faster model if available
        print("Initializing FCoref...")
        model = FCoref()
        
        print("Running prediction...")
        result = model.predict([text])
        
        print(f"Result: {result}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def try_alternative_ontonotes_source():
    """Try to find alternative OntoNotes evaluation data."""
    print("\n=== Looking for Alternative OntoNotes Source ===")
    
    # Try the GitHub preprocessed version
    github_url = "https://raw.githubusercontent.com/ontonotes/conll-formatted-ontonotes-5.0/master/README.md"
    
    try:
        response = requests.get(github_url, timeout=10)
        if response.status_code == 200:
            print("Found GitHub repository with CoNLL formatted OntoNotes!")
            print("Repository: https://github.com/ontonotes/conll-formatted-ontonotes-5.0")
            return True
        else:
            print(f"GitHub source not accessible: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error accessing GitHub source: {e}")
        return False

def check_evaluation_alternatives():
    """Check what evaluation frameworks are available."""
    print("\n=== Evaluation Framework Options ===")
    
    print("1. Direct fastcoref evaluation on custom text")
    print("2. Use CoNLL evaluation script if available")
    print("3. Create simple evaluation metrics manually")
    
    # For now, let's focus on getting fastcoref working
    return try_simple_fastcoref_test()

if __name__ == "__main__":
    # Start with simple test
    fastcoref_works = check_evaluation_alternatives()
    
    if fastcoref_works:
        print("\n=== SUCCESS: Fastcoref is working! ===")
        print("Next step: Set up evaluation data")
    else:
        print("\n=== ISSUE: Need to debug fastcoref setup ===")
        
    # Check for alternative data sources
    alt_source = try_alternative_ontonotes_source()
    
    if alt_source:
        print("Found alternative OntoNotes source on GitHub!")
    
    print("\n=== Recommendations ===")
    print("1. Focus on getting fastcoref working with simple examples first")
    print("2. Then set up evaluation with either:")
    print("   - GitHub OntoNotes CoNLL format")
    print("   - Manual evaluation on GAP dataset (our primary target)")
    print("   - Custom challenging examples")