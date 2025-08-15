#!/usr/bin/env python3
"""Test fastcoref built-in OntoNotes evaluation capabilities."""

import os
from fastcoref import FCoref, LingMessCoref

def test_fastcoref_ontonotes_evaluation():
    """Test if fastcoref has built-in OntoNotes evaluation capabilities."""
    
    print("=== Testing Fastcoref OntoNotes Evaluation ===")
    
    # Check if FCoref has evaluation methods
    model = FCoref()
    
    print(f"FCoref methods: {[method for method in dir(model) if not method.startswith('_')]}")
    
    # Check for any evaluation or dataset methods
    evaluation_methods = [method for method in dir(model) if 'eval' in method.lower() or 'ontonotes' in method.lower() or 'dataset' in method.lower()]
    
    if evaluation_methods:
        print(f"Found evaluation methods: {evaluation_methods}")
    else:
        print("No obvious evaluation methods found")
    
    # Check if there are any class methods for dataset handling
    class_methods = [method for method in dir(FCoref) if not method.startswith('_')]
    print(f"FCoref class methods: {class_methods}")

def check_fastcoref_module_structure():
    """Check the fastcoref module structure for evaluation capabilities."""
    
    print("\n=== Checking Fastcoref Module Structure ===")
    
    try:
        import fastcoref
        print(f"Fastcoref module contents: {dir(fastcoref)}")
        
        # Check for any evaluation submodules
        if hasattr(fastcoref, 'evaluation'):
            print(f"Found evaluation module: {dir(fastcoref.evaluation)}")
        
        if hasattr(fastcoref, 'datasets'):
            print(f"Found datasets module: {dir(fastcoref.datasets)}")
            
        # Check the main models
        from fastcoref import FCoref, LingMessCoref
        print(f"Available models: FCoref, LingMessCoref")
        
    except Exception as e:
        print(f"Error exploring fastcoref module: {e}")

def look_for_ontonotes_evaluation_scripts():
    """Look for any OntoNotes evaluation scripts or examples."""
    
    print("\n=== Looking for OntoNotes Evaluation Examples ===")
    
    # Check if there are any examples or scripts in the installed package
    try:
        import fastcoref
        package_path = os.path.dirname(fastcoref.__file__)
        print(f"Fastcoref package path: {package_path}")
        
        # List contents of the package directory
        if os.path.exists(package_path):
            contents = os.listdir(package_path)
            print(f"Package contents: {contents}")
            
            # Look for evaluation or example files
            eval_files = [f for f in contents if 'eval' in f.lower() or 'example' in f.lower() or 'ontonotes' in f.lower()]
            if eval_files:
                print(f"Found evaluation/example files: {eval_files}")
        
    except Exception as e:
        print(f"Error exploring package structure: {e}")

def test_alternative_approaches():
    """Test alternative approaches to OntoNotes evaluation."""
    
    print("\n=== Alternative Approaches ===")
    
    print("Option 1: Use our custom test cases as proxy for OntoNotes performance")
    print("- We already have challenging cases that represent OntoNotes-style problems")
    print("- Can validate against published fastcoref results on OntoNotes")
    
    print("\nOption 2: Use the GAP dataset (our primary target)")  
    print("- More relevant to our research goals")
    print("- Publicly available")
    print("- Directly tests multi-entity ambiguous pronouns")
    
    print("\nOption 3: Create OntoNotes-style evaluation from preprocessed data")
    print("- Use publicly available preprocessed OntoNotes subsets")
    print("- Focus on coreference-heavy examples")
    
    print("\nRecommendation: Start with GAP dataset + our challenging test cases")
    print("- This gives us solid baseline for our specific research target")
    print("- We can reference published OntoNotes results for comparison")

if __name__ == "__main__":
    test_fastcoref_ontonotes_evaluation()
    check_fastcoref_module_structure()
    look_for_ontonotes_evaluation_scripts()
    test_alternative_approaches()