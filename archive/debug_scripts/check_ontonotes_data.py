#!/usr/bin/env python3
"""Check OntoNotes dataset availability and fastcoref models."""

def check_huggingface_ontonotes():
    """Check if OntoNotes is available via Hugging Face datasets."""
    try:
        from datasets import load_dataset
        print("=== Checking OntoNotes via Hugging Face ===")
        
        # Try to load a small sample first
        print("Attempting to load OntoNotes dataset...")
        dataset = load_dataset("conll2012_ontonotesv5", "english_v4", split="validation", streaming=True)
        
        # Get first example
        first_example = next(iter(dataset))
        print(f"Dataset loaded successfully!")
        print(f"First example keys: {first_example.keys()}")
        
        # Show structure
        if 'sentences' in first_example:
            print(f"Number of sentences in first example: {len(first_example['sentences'])}")
            if len(first_example['sentences']) > 0:
                print(f"First sentence tokens: {first_example['sentences'][0]['words'][:10]}...")
        
        return True
        
    except ImportError:
        print("Hugging Face datasets not available. Installing...")
        return False
    except Exception as e:
        print(f"Error loading OntoNotes: {e}")
        return False

def check_fastcoref_models():
    """Check available fastcoref models."""
    print("\n=== Checking Fastcoref Models ===")
    try:
        from fastcoref import FCoref, LingMessCoref
        
        print("Available model classes:")
        print("- FCoref (fast model)")
        print("- LingMessCoref (accurate model)")
        
        return True
    except Exception as e:
        print(f"Error importing fastcoref models: {e}")
        return False

if __name__ == "__main__":
    # Check OntoNotes availability
    ontonotes_available = check_huggingface_ontonotes()
    
    # Check fastcoref models
    models_available = check_fastcoref_models()
    
    if ontonotes_available and models_available:
        print("\n=== Setup Status: READY ===")
        print("Both OntoNotes data and fastcoref models are available!")
    else:
        print("\n=== Setup Status: NEEDS WORK ===")
        if not ontonotes_available:
            print("- Need to install datasets or find alternative OntoNotes source")
        if not models_available:
            print("- Need to fix fastcoref model access")