#!/usr/bin/env python3
"""
Demo and Tutorial: EntityExtractor
==================================

This script demonstrates how to use the EntityExtractor class for named entity
recognition with spaCy + DistilBERT fallback.

Run with: uv run python demo_entity_extractor.py
"""

from flexmind.core.extractors.entities import EntityExtractor
from flexmind.core.preprocessing.text import TextPreprocessor
import time


def demo_basic_extraction():
    """Demo: Basic entity extraction from simple text."""
    print("=" * 60)
    print("DEMO 1: Basic Entity Extraction")
    print("=" * 60)
    
    extractor = EntityExtractor()
    
    sample_texts = [
        "Alice Johnson works at Microsoft in San Francisco since 2024.",
        "President Biden met with Elon Musk in Washington D.C. on January 15th.",
        "The AI researcher Dr. Yann LeCun leads FAIR at Meta in Menlo Park.",
        "OpenAI released GPT-4 which was trained on massive datasets."
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"Text {i}: {text}")
        
        start_time = time.time()
        entities = extractor.extract(text)
        extraction_time = time.time() - start_time
        
        print(f"Extracted {len(entities)} entities in {extraction_time:.3f}s:")
        
        if entities:
            for entity in entities:
                print(f"  {entity}")
        else:
            print("  No entities found")
        
        print("-" * 40)
    
    return entities


def demo_fallback_behavior():
    """Demo: How DistilBERT fallback works with complex text."""
    print("\n" + "=" * 60)
    print("DEMO 2: Fallback Behavior for Complex Text")
    print("=" * 60)
    
    # Test with and without fallback
    extractor_with_fallback = EntityExtractor(use_fallback=True)
    extractor_no_fallback = EntityExtractor(use_fallback=False)
    
    complex_text = """
    The startup's CTO, Dr. Sarah Chen-Williams, previously worked at DeepMind 
    on reinforcement learning algorithms. She's now building an AI safety 
    framework at her company NeuralGuard in Palo Alto.
    """
    
    print(f"Complex text:\n{complex_text.strip()}\n")
    
    print("WITH fallback (spaCy + DistilBERT):")
    start_time = time.time()
    entities_with = extractor_with_fallback.extract(complex_text)
    time_with = time.time() - start_time
    
    for entity in entities_with:
        print(f"  {entity}")
    print(f"  Time: {time_with:.3f}s, Count: {len(entities_with)}")
    
    print("\nWITHOUT fallback (spaCy only):")
    start_time = time.time()
    entities_without = extractor_no_fallback.extract(complex_text)
    time_without = time.time() - start_time
    
    for entity in entities_without:
        print(f"  {entity}")
    print(f"  Time: {time_without:.3f}s, Count: {len(entities_without)}")
    
    print(f"\nFallback added {len(entities_with) - len(entities_without)} entities")
    print(f"Processing time difference: {time_with - time_without:.3f}s")


def demo_with_preprocessing():
    """Demo: Entity extraction on preprocessed text chunks."""
    print("\n" + "=" * 60)
    print("DEMO 3: Entity Extraction on Preprocessed Chunks")
    print("=" * 60)
    
    preprocessor = TextPreprocessor()
    extractor = EntityExtractor()
    
    dialog_text = """
    Alice: I just got hired at Anthropic in San Francisco!
    Bob: Congrats! When do you start?
    Alice: Next Monday. I'll be working with Claude's safety team.
    Bob: That's amazing. Did you meet Dario Amodei during interviews?
    Alice: Yes, he's brilliant. We discussed constitutional AI approaches.
    """
    
    print(f"Dialog text:\n{dialog_text.strip()}\n")
    
    # First preprocess into chunks
    chunks = preprocessor.process(dialog_text, text_type='dialog')
    print(f"Preprocessed into {len(chunks)} chunks\n")
    
    # Extract entities from each chunk
    all_entities = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i} entities:")
        entities = extractor.extract(chunk.text)
        
        if entities:
            for entity in entities:
                print(f"  {entity}")
                all_entities.append(entity.to_tuple())
        else:
            print("  No entities found")
        print()
    
    # Deduplicate across all chunks
    unique_entities = list(set(all_entities))
    print(f"Total unique entities across all chunks: {len(unique_entities)}")
    for entity_text, entity_label in unique_entities:
        print(f"  '{entity_text}' → {entity_label}")


def demo_confidence_filtering():
    """Demo: How confidence filtering works."""
    print("\n" + "=" * 60)
    print("DEMO 4: Confidence Filtering")
    print("=" * 60)
    
    # Test different confidence thresholds
    thresholds = [0.5, 0.75, 0.9]
    
    ambiguous_text = "The meeting with the CEO about the new project in the building was scheduled."
    
    print(f"Ambiguous text: {ambiguous_text}\n")
    
    for threshold in thresholds:
        extractor = EntityExtractor(confidence_threshold=threshold)
        entities = extractor.extract(ambiguous_text)
        
        print(f"Confidence threshold {threshold}: {len(entities)} entities")
        for entity in entities:
            print(f"  {entity}")
        print()


def demo_interactive_extraction():
    """Interactive mode for testing your own text."""
    print("\n" + "=" * 60)
    print("DEMO 5: Interactive Entity Extraction")
    print("=" * 60)
    
    extractor = EntityExtractor()
    
    print("Enter text to extract entities from (or 'quit' to exit):")
    print("Try examples like:")
    print("- 'Apple CEO Tim Cook visited Paris last week'")
    print("- 'The AI conference in London featured speakers from Google and Meta'")
    print("-" * 40)
    
    while True:
        try:
            text = input("\n> Your text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
                
            if not text:
                continue
            
            start_time = time.time()
            entities = extractor.extract(text)
            extraction_time = time.time() - start_time
            
            print(f"\nFound {len(entities)} entities in {extraction_time:.3f}s:")
            
            if entities:
                # Show individual entities with their metadata
                for entity in entities:
                    print(f"  {entity}")
                
                # Also show grouped summary
                print("\n  Summary by type:")
                entity_groups = {}
                for entity in entities:
                    if entity.label not in entity_groups:
                        entity_groups[entity.label] = []
                    entity_groups[entity.label].append(entity.text)
                
                for label, texts in entity_groups.items():
                    print(f"    {label}: {', '.join(texts)}")
            else:
                print("  No entities found")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThanks for trying the EntityExtractor demo!")


def tutorial_configuration():
    """Tutorial: How to configure the EntityExtractor."""
    print("\n" + "=" * 60)
    print("TUTORIAL: EntityExtractor Configuration")
    print("=" * 60)
    
    print("""
EntityExtractor Configuration Options:

1. confidence_threshold (float, default=0.75):
   - Minimum confidence score for including entities
   - Higher = fewer, higher-quality entities
   - Lower = more entities, potentially lower quality

2. use_fallback (bool, default=True):
   - Whether to use DistilBERT fallback for complex cases
   - True = better coverage, slower processing
   - False = faster processing, may miss complex entities

Performance Characteristics:
- spaCy: ~50,000-100,000 tokens/second (primary)
- DistilBERT: Used only when spaCy coverage is insufficient
- Automatic fallback triggers on: low entity density, complex patterns, zero entities

Entity Types Supported:
- PERSON: People's names
- ORG: Organizations, companies
- GPE: Geopolitical entities (cities, countries)
- DATE: Dates and times
- MONEY: Monetary amounts
- MISC: Miscellaneous entities
""")


def main():
    """Run all demos and tutorials."""
    print("FlexMind EntityExtractor Demo")
    print("=============================")
    print("This demo shows how to use the EntityExtractor for:")
    print("- Basic named entity recognition")
    print("- Fallback behavior with complex text")
    print("- Integration with text preprocessing")
    print("- Confidence filtering")
    print("- Interactive testing")
    
    try:
        # Run demos
        demo_basic_extraction()
        demo_fallback_behavior()
        demo_with_preprocessing()
        demo_confidence_filtering()
        tutorial_configuration()
        
        # Interactive mode
        print("\n" + "=" * 60)
        response = input("Would you like to try interactive mode? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            demo_interactive_extraction()
            
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!")
    except Exception as e:
        print(f"\nDemo error: {e}")
        print("Make sure you're running with: uv run python demo_entity_extractor.py")


if __name__ == "__main__":
    main()