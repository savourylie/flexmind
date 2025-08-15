#!/usr/bin/env python3
"""
Demo and Tutorial: TextPreprocessor
===================================

This script demonstrates how to use the TextPreprocessor class for different
types of content (dialog, documents, general text).

Run with: uv run python demo_text_preprocessor.py
"""

from flexmind.core.preprocessing.text import TextPreprocessor


def demo_dialog_processing():
    """Demo: Processing dialog with sliding window chunking."""
    print("=" * 60)
    print("DEMO 1: Dialog Processing with Sliding Window")
    print("=" * 60)
    
    preprocessor = TextPreprocessor(window_size=3)  # Smaller window for demo
    
    dialog_text = """
    Alice: Hey Bob, I just started working at OpenAI!
    Bob: That's amazing! When did you move to San Francisco?
    Alice: Last month. I'm working on GPT-5 with the alignment team.
    Bob: How do you like the team culture there?
    Alice: Everyone is brilliant. My manager Sarah is fantastic.
    Bob: What's your specific role?
    Alice: I'm a senior research scientist focusing on safety research.
    Bob: That sounds like important work!
    Alice: Yeah, we're trying to solve alignment before AGI arrives.
    """
    
    print(f"Input dialog:\n{dialog_text.strip()}\n")
    
    chunks = preprocessor.process(dialog_text, text_type='dialog')
    
    print(f"Generated {len(chunks)} chunks with sliding window size 3:")
    print("-" * 40)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")
        print(f"Full text: {chunk.text}")
        print(f"Metadata: {chunk.metadata}")
        print("-" * 40)
    
    return chunks


def demo_document_processing():
    """Demo: Processing documents with sentence-based chunking."""
    print("\n" + "=" * 60)
    print("DEMO 2: Document Processing with Sentence Chunking")
    print("=" * 60)
    
    preprocessor = TextPreprocessor(max_tokens=50)  # Small token limit for demo
    
    document_text = """
    The FlexMind knowledge graph memory system is designed for speed and efficiency.
    It uses spaCy for named entity recognition at 100,000 tokens per second on CPU.
    The system also employs DistilBERT as a fallback for complex entity recognition cases.
    Neo4j serves as the graph database for storing entities and relationships with temporal tracking.
    ChromaDB provides vector storage for unstructured content retrieval and semantic search.
    The hybrid architecture combines structured knowledge graphs with vector embeddings.
    This approach enables both precise fact retrieval and semantic similarity matching.
    Performance benchmarks show sub-second query responses on standard hardware.
    """
    
    print(f"Input document:\n{document_text.strip()}\n")
    
    chunks = preprocessor.process(document_text, text_type='document')
    
    print(f"Generated {len(chunks)} chunks with max 50 tokens each:")
    print("-" * 40)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")
        print(f"Full text: {chunk.text}")
        print("-" * 40)
    
    return chunks


def demo_interactive_mode():
    """Interactive mode for testing your own text."""
    print("\n" + "=" * 60)
    print("DEMO 3: Interactive Mode - Test Your Own Text!")
    print("=" * 60)
    
    preprocessor = TextPreprocessor()
    
    print("Enter your text (or 'quit' to exit):")
    print("Tip: For dialog, use 'Speaker: message' format")
    print("     For documents, just write regular paragraphs")
    print("-" * 40)
    
    while True:
        try:
            text = input("\n> Your text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
                
            if not text:
                continue
            
            # Auto-detect text type based on content
            text_type = 'dialog' if ':' in text and any(line.strip().count(':') == 1 for line in text.split('\n')) else 'document'
            
            print(f"\nDetected type: {text_type}")
            
            chunks = preprocessor.process(text, text_type=text_type)
            
            if chunks:
                print(f"Generated {len(chunks)} chunk(s):")
                for i, chunk in enumerate(chunks, 1):
                    print(f"\nChunk {i}: {chunk}")
                    if len(chunk.text) > 200:
                        print(f"Full text: {chunk.text[:200]}...")
                    else:
                        print(f"Full text: {chunk.text}")
            else:
                print("No chunks generated (empty or invalid text)")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThanks for trying the TextPreprocessor demo!")


def tutorial_configuration():
    """Tutorial: How to configure the TextPreprocessor."""
    print("\n" + "=" * 60)
    print("TUTORIAL: TextPreprocessor Configuration")
    print("=" * 60)
    
    print("""
TextPreprocessor Configuration Options:

1. window_size (int, default=5): 
   - For dialog processing: number of conversation turns per chunk
   - Larger = more context, fewer chunks
   - Smaller = less context, more chunks

2. max_tokens (int, default=512):
   - For document processing: maximum tokens per chunk
   - Respects sentence boundaries
   - Prevents chunks from being too large for downstream models

Example configurations:
""")
    
    # Demo different configurations
    configs = [
        {"window_size": 3, "max_tokens": 100, "use_case": "Fast processing, less context"},
        {"window_size": 7, "max_tokens": 1024, "use_case": "Rich context, slower processing"},
        {"window_size": 5, "max_tokens": 512, "use_case": "Balanced (default)"}
    ]
    
    sample_text = "Alice: Hi! Bob: Hello! Alice: How are you? Bob: Good! Alice: Great!"
    
    for config in configs:
        print(f"Config: {config}")
        preprocessor = TextPreprocessor(**{k: v for k, v in config.items() if k != 'use_case'})
        chunks = preprocessor.process(sample_text, text_type='dialog')
        print(f"  → Generated {len(chunks)} chunks")
        print()


def main():
    """Run all demos and tutorials."""
    print("FlexMind TextPreprocessor Demo")
    print("==============================")
    print("This demo shows how to use the TextPreprocessor for:")
    print("- Dialog processing with sliding windows")
    print("- Document processing with sentence chunking")
    print("- Interactive testing with your own text")
    print("- Configuration options")
    
    try:
        # Run demos
        demo_dialog_processing()
        demo_document_processing()
        tutorial_configuration()
        
        # Interactive mode
        print("\n" + "=" * 60)
        response = input("Would you like to try interactive mode? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            demo_interactive_mode()
            
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!")
    except Exception as e:
        print(f"\nDemo error: {e}")
        print("Make sure you're running with: uv run python demo_text_preprocessor.py")


if __name__ == "__main__":
    main()