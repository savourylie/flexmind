from flexmind.chunking import CorefSafeChunker

def debug_chunker_behavior():
    chunker = CorefSafeChunker(target_size=15, max_size=25, overlap_sentences=1)
    
    # Test the specific case from the failing test
    text = "Dr. Johnson works at the hospital. The building is very tall. He treats many patients there."
    
    print("=== Micro Chunker Test ===")
    print(f"Text: {text}")
    chunks = chunker.chunk(text)
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: '{chunk.text}'")
        print(f"  Token count: {chunk.metadata['token_count']}")
        print(f"  Has entities: {chunk.metadata['has_entities']}")
        print(f"  Pronoun density: {chunk.metadata['pronoun_density']:.3f}")
        print(f"  Starts with Dr. Johnson: {chunk.text.strip().startswith('Dr. Johnson')}")
        print(f"  Contains 'He treats': {'He treats many patients' in chunk.text}")
        print()
    
    # Check what the test is looking for
    print("=== Analyzing chunks that contain 'He treats' ===")
    for i, chunk in enumerate(chunks):
        if "He treats many patients" in chunk.text:
            print(f"Chunk {i} contains 'He treats many patients':")
            sentences = chunker._split_into_sentences(chunk.text)
            print(f"  Sentences: {sentences}")
            print(f"  Sentence count: {len(sentences)}")
            print(f"  Expected ≥3 for start rule success: {len(sentences) >= 3}")

if __name__ == "__main__":
    debug_chunker_behavior()