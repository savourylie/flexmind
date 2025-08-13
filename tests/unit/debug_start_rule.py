#!/usr/bin/env python3

from flexmind.chunking import CorefSafeChunker

def debug_start_rule_issue():
    chunker = CorefSafeChunker(target_size=50, max_size=100)
    
    text = """Dr. Sarah Martinez led the research team at the prestigious Stanford AI Lab. The team had been working on breakthrough neural architecture designs for months. She discovered that attention mechanisms could be significantly improved by incorporating memory-like structures. He colleague Dr. James Liu was skeptical at first. It turned out to be one of the most important discoveries in modern AI."""
    
    print("=== Debugging Start Rule Issue ===")
    chunks = chunker.chunk(text)
    
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}: '{chunk.text}'")
        print(f"  Tokens: {chunk.metadata['token_count']}")
        
        # Check first sentence of chunk
        sentences = chunker._split_into_sentences(chunk.text)
        if sentences:
            first_sentence = sentences[0]
            print(f"  First sentence: '{first_sentence}'")
            print(f"  First sentence has entities: {chunker._has_named_entities(first_sentence)}")
            print(f"  First sentence has pronouns: {chunker._contains_pronouns(first_sentence)}")
            
            # Check if this should trigger start rule
            if i > 0:  # Not first chunk
                should_trigger = not chunker._has_named_entities(first_sentence) and chunker._contains_pronouns(first_sentence)
                print(f"  Should trigger start rule: {should_trigger}")
                
                if should_trigger:
                    print(f"  🚨 BUG: Start rule should have triggered but didn't!")
                    
                    # Debug what the start rule is finding
                    all_sentences = chunker._split_into_sentences(text)
                    previous_sentences = chunker._get_previous_sentences_with_entities(all_sentences, i, chunks)
                    print(f"  Previous sentences found: {previous_sentences}")
                    
                    # Check what entities are in the current chunk
                    current_entities = []
                    for sent in sentences:
                        if chunker._has_named_entities(sent):
                            doc = chunker.nlp(sent)
                            entities = [ent.text for ent in doc.ents if ent.label_ not in {'DATE', 'TIME', 'CARDINAL', 'ORDINAL'}]
                            current_entities.extend(entities)
                    print(f"  Entities in current chunk: {current_entities}")

if __name__ == "__main__":
    debug_start_rule_issue()