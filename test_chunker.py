#!/usr/bin/env python3
"""Test script for CorefSafeChunker with Adrian Mallory Finch text data."""

from flexmind.chunking.coref_chunker import CorefSafeChunker
import json


def main():
    # Read the text data
    with open('datasets/backstories/adrian_mallory_finch.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("=== Original Text Info ===")
    print(f"Total characters: {len(text)}")
    print(f"Total words (approx): {len(text.split())}")
    print("\n" + "="*50 + "\n")
    
    # Initialize chunker with default settings
    chunker = CorefSafeChunker(target_size=350, max_size=600, overlap_sentences=2)
    
    # Chunk the text
    chunks = chunker.chunk(text)
    
    print(f"=== Chunking Results ===")
    print(f"Number of chunks created: {len(chunks)}")
    print("\n")
    
    # Analyze and display each chunk
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(f"Text: {chunk.text[:100]}{'...' if len(chunk.text) > 100 else ''}")
        print(f"Start index: {chunk.start_idx}")
        print(f"End index: {chunk.end_idx}")
        print(f"Length: {len(chunk.text)} characters")
        print("Metadata:")
        for key, value in chunk.metadata.items():
            print(f"  {key}: {value}")
        print()
    
    # Show the data schema structure
    print("=== Data Schema ===")
    if chunks:
        example_chunk = chunks[0]
        schema = {
            "Chunk": {
                "text": "str - The actual text content",
                "start_idx": "int - Starting character index in original text", 
                "end_idx": "int - Ending character index in original text",
                "metadata": {
                    "chunk_id": "int - Sequential chunk identifier",
                    "token_count": "int - Number of tokens in chunk",
                    "sentence_count": "int - Number of sentences in chunk", 
                    "has_entities": "bool - Whether chunk contains named entities",
                    "pronoun_density": "float - Ratio of pronouns to total tokens"
                }
            }
        }
        
        print(json.dumps(schema, indent=2))
        
    # Summary statistics
    total_tokens = sum(chunk.metadata['token_count'] for chunk in chunks)
    avg_tokens = total_tokens / len(chunks) if chunks else 0
    chunks_with_entities = sum(1 for chunk in chunks if chunk.metadata['has_entities'])
    avg_pronoun_density = sum(chunk.metadata['pronoun_density'] for chunk in chunks) / len(chunks) if chunks else 0
    
    print("\n=== Summary Statistics ===")
    print(f"Total tokens across all chunks: {total_tokens}")
    print(f"Average tokens per chunk: {avg_tokens:.1f}")
    print(f"Chunks with named entities: {chunks_with_entities}/{len(chunks)}")
    print(f"Average pronoun density: {avg_pronoun_density:.3f}")


if __name__ == "__main__":
    main()