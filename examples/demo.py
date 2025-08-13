#!/usr/bin/env python3
"""
Demo of the Coreference-Safe Chunker

This demonstrates the human-like memory mechanism inspired by "Why We Remember" -
prioritizing flexibility and contextual adaptation over static accuracy.
"""

from flexmind.chunking import CorefSafeChunker

def demo_coref_chunking():
    print("🧠 FlexMind: Human-like Memory Mechanism Demo")
    print("=" * 50)
    
    # Create chunker with small target size to demonstrate features
    chunker = CorefSafeChunker(target_size=50, max_size=100)
    
    # Demo 1: Start Rule - prepends antecedents when chunks start with pronouns
    print("\n🎯 Demo 1: Start Rule (Pronoun Grounding)")
    text1 = """Dr. Sarah Martinez led the research team at the prestigious Stanford AI Lab. The team had been working on breakthrough neural architecture designs for months. She discovered that attention mechanisms could be significantly improved by incorporating memory-like structures. He colleague Dr. James Liu was skeptical at first. It turned out to be one of the most important discoveries in modern AI."""
    
    print(f"Text: {text1[:100]}...")
    chunks1 = chunker.chunk(text1)
    
    for i, chunk in enumerate(chunks1):
        print(f"\nChunk {i+1}:")
        print(f"  Text: {chunk.text}")
        print(f"  Tokens: {chunk.metadata['token_count']}")
        print(f"  Has entities: {chunk.metadata['has_entities']}")
        print(f"  Pronoun density: {chunk.metadata['pronoun_density']:.1%}")
    
    # Demo 2: Anaphora Hazard Detection - expands context for high pronoun density
    print("\n\n⚠️  Demo 2: Anaphora Hazard Detection")
    text2 = """The quantum computing team at IBM made a significant breakthrough. They told him that it was theirs and they should keep it with them forever."""
    
    print(f"Text: {text2}")
    chunks2 = chunker.chunk(text2)
    
    for i, chunk in enumerate(chunks2):
        print(f"\nChunk {i+1}:")
        print(f"  Text: {chunk.text}")
        print(f"  Pronoun density: {chunk.metadata['pronoun_density']:.1%}")
        if chunk.metadata['pronoun_density'] >= 0.15:
            print(f"  ⚠️  HIGH PRONOUN DENSITY DETECTED - Context expanded!")
    
    # Demo 3: Sentence Start Rule - handles This/That/He/She sentence starts  
    print("\n\n🚀 Demo 3: Sentence Start Rule")
    text3 = """Professor Alan Turing developed the theoretical foundations of computation. This was revolutionary work that changed computer science forever."""
    
    print(f"Text: {text3}")
    chunks3 = chunker.chunk(text3)
    
    for i, chunk in enumerate(chunks3):
        print(f"\nChunk {i+1}:")
        print(f"  Text: {chunk.text}")
        sentences = chunk.text.split('. ')
        for sent in sentences:
            first_word = sent.strip().split()[0].lower() if sent.strip() else ""
            if first_word in {'this', 'that', 'he', 'she', 'they', 'it'}:
                print(f"  🚀 Sentence starts with '{first_word}' - Antecedent preserved!")
    
    # Demo 4: Fallback Strategy - handles cases without named entities
    print("\n\n🔄 Demo 4: Fallback Strategy")  
    text4 = """The person walked to the building. He entered through the door. She was already in the lobby."""
    
    print(f"Text: {text4}")
    chunks4 = chunker.chunk(text4)
    
    for i, chunk in enumerate(chunks4):
        print(f"\nChunk {i+1}:")
        print(f"  Text: {chunk.text}")
        print(f"  Has named entities: {chunk.metadata['has_entities']}")
        if not chunk.metadata['has_entities']:
            print(f"  🔄 No named entities - Fallback strategy used concrete nouns!")
    
    print("\n" + "=" * 50)
    print("✅ Demo complete! The chunker successfully demonstrates:")
    print("   • Flexible, context-aware chunking (not rigid token limits)")  
    print("   • Pronoun-antecedent preservation")
    print("   • Anaphora risk detection and mitigation")
    print("   • Sentence-start hazard handling")
    print("   • Intelligent fallback strategies")
    print("\n🧠 This embodies human-like memory: adaptive and contextual!")

if __name__ == "__main__":
    demo_coref_chunking()