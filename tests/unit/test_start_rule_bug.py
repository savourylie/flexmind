#!/usr/bin/env python3

import pytest
from flexmind.chunking import CorefSafeChunker

def test_start_rule_looks_back_for_correct_antecedent():
    """Test that start rule finds the correct antecedent, not just any entity"""
    chunker = CorefSafeChunker(target_size=50, max_size=100)
    
    # This text should chunk such that "She" needs "Dr. Sarah Martinez" as antecedent
    text = """Dr. Sarah Martinez led the research team at the prestigious Stanford AI Lab. The team had been working on breakthrough neural architecture designs for months. She discovered that attention mechanisms could be significantly improved by incorporating memory-like structures. He colleague Dr. James Liu was skeptical at first. It turned out to be one of the most important discoveries in modern AI."""
    
    chunks = chunker.chunk(text)
    
    # Find chunk that contains "She discovered" (should NOT start with it due to start rule)
    she_chunk = None
    for chunk in chunks:
        if "She discovered" in chunk.text:
            she_chunk = chunk
            break
    
    assert she_chunk is not None, "Should find chunk containing 'She discovered'"
    
    # The start rule should prevent chunks from starting with "She" by prepending antecedents
    sentences = chunker._split_into_sentences(she_chunk.text)
    assert not sentences[0].startswith("She"), \
        f"Start rule should prevent chunk from starting with pronoun 'She'. Got first sentence: '{sentences[0]}'"
    
    # The chunk should contain "Dr. Sarah Martinez" as the antecedent for "She"
    assert "Dr. Sarah Martinez" in she_chunk.text, \
        f"Start rule should include 'Dr. Sarah Martinez' as antecedent for 'She'. Got: {she_chunk.text}"

if __name__ == "__main__":
    test_start_rule_looks_back_for_correct_antecedent()