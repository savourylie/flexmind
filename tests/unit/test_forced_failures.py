"""
More aggressive tests designed to definitely expose the sentence mapping and rule application bugs.
These tests use scenarios that should absolutely fail with the current implementation.
"""

import pytest
from flexmind.chunking import CorefSafeChunker


class TestDefiniteSentenceMappingFailures:
    """Tests that should absolutely expose sentence mapping failures"""
    
    def test_sentence_lookup_with_modified_chunks_after_rules(self):
        """This should definitely fail - chunks modified by rules break sentence lookup"""
        chunker = CorefSafeChunker(target_size=30, max_size=60, overlap_sentences=1)
        
        # Create text where sentence will be prepended by start rule
        text = "Dr. Smith examined the patient carefully in the morning. He found several critical issues."
        
        chunks = chunker.chunk(text)
        original_sentences = chunker._split_into_sentences(text)
        
        # Find chunk that was modified by start rule (should contain both sentences)
        modified_chunk = None
        for chunk in chunks:
            if "He found several" in chunk.text and "Dr. Smith examined" in chunk.text:
                modified_chunk = chunk
                break
        
        if modified_chunk:
            # The start rule should have prepended "Dr. Smith examined" to "He found several"
            chunk_sentences = chunker._split_into_sentences(modified_chunk.text)
            
            # Now, try to use this modified chunk in the sentence lookup mechanism
            # This should break because the chunk structure no longer matches original sentences
            
            # Get the chunk index
            chunk_idx = None
            for i, c in enumerate(chunks):
                if c.text == modified_chunk.text:
                    chunk_idx = i
                    break
            
            if chunk_idx is not None:
                # Try to get previous sentences - this will use the broken lookup
                try:
                    previous = chunker._get_previous_sentences_with_entities(
                        original_sentences, chunk_idx, chunks, max_sentences=1
                    )
                    
                    # The issue: it tries to find the first sentence of the modified chunk
                    # in the original sentence list, but the first sentence might be a prepended one
                    first_chunk_sentence = chunk_sentences[0]
                    
                    # Check if this sentence actually exists at the expected position
                    expected_position = None
                    for i, orig_sent in enumerate(original_sentences):
                        if orig_sent.strip() == first_chunk_sentence.strip():
                            expected_position = i
                            break
                    
                    # This should expose the bug
                    assert expected_position is not None, \
                        f"Modified chunk's first sentence '{first_chunk_sentence}' " \
                        f"not found in original sentences {original_sentences}. " \
                        f"This proves sentence mapping breaks after rule modifications."
                    
                except Exception as e:
                    # If it throws an exception, that also proves the mechanism is broken
                    pytest.fail(f"Sentence lookup mechanism failed with error: {e}")
    
    def test_duplicate_sentences_wrong_position_lookup(self):
        """This should definitely fail - identical sentences cause wrong position lookup"""
        chunker = CorefSafeChunker(target_size=40, max_size=80)
        
        # Create text with identical sentences at different positions
        text = "The dog ran fast. The cat sat quietly. The dog ran fast. He was very tired. The dog ran fast."
        
        sentences = chunker._split_into_sentences(text)
        chunks = chunker.chunk(text)
        
        # "The dog ran fast" appears at positions 0, 2, and 4
        # "He was very tired" should be at position 3
        
        # Find the actual position of "He was very tired"
        he_position = None
        for i, sent in enumerate(sentences):
            if "He was very tired" in sent:
                he_position = i
                break
        
        assert he_position == 3, f"Test setup wrong - 'He was very tired' should be at position 3, got {he_position}"
        
        # Find chunk containing "He was very tired"
        he_chunk_idx = None
        for i, chunk in enumerate(chunks):
            if "He was very tired" in chunk.text:
                he_chunk_idx = i
                break
        
        if he_chunk_idx is not None:
            # The sentence lookup should find the sentence at position 2 (the third "The dog ran fast")
            # as the immediate predecessor to "He was very tired" at position 3
            previous = chunker._get_previous_sentences_with_entities(
                sentences, he_chunk_idx, chunks, max_sentences=1
            )
            
            if previous:
                # The bug: it will find the FIRST "The dog ran fast" at position 0
                # instead of the correct one at position 2
                found_sentence = previous[0]
                expected_sentence = sentences[2]  # The third "The dog ran fast"
                
                # This test exposes that it finds the wrong identical sentence
                assert found_sentence == expected_sentence, \
                    f"Wrong identical sentence found. Expected: '{expected_sentence}' (position 2), " \
                    f"Got: '{found_sentence}'. This proves the lookup finds the first occurrence, not the correct one."
    
    def test_rough_estimate_fallback_produces_impossible_results(self):
        """This should definitely fail - rough estimate can be beyond array bounds"""
        chunker = CorefSafeChunker(target_size=25, max_size=50)
        
        # Create short text where rough estimate exceeds sentence count
        text = "Alice works here. Bob helps Alice. She is very busy."
        
        sentences = chunker._split_into_sentences(text)
        chunks = chunker.chunk(text)
        
        # Should have 3 sentences total
        assert len(sentences) == 3
        
        # Find chunk with "She is very busy"
        she_chunk_idx = None
        for i, chunk in enumerate(chunks):
            if "She is very busy" in chunk.text:
                she_chunk_idx = i
                break
        
        if she_chunk_idx is not None and she_chunk_idx >= 2:
            # If chunk_idx >= 2, then rough estimate = chunk_idx * 2 >= 4
            # But we only have 3 sentences (indices 0, 1, 2)
            rough_estimate = she_chunk_idx * 2
            
            # This should fail - estimate exceeds array bounds
            assert rough_estimate < len(sentences), \
                f"Rough estimate fallback produces impossible result: " \
                f"chunk_idx={she_chunk_idx} * 2 = {rough_estimate}, " \
                f"but only have {len(sentences)} sentences (max index {len(sentences)-1}). " \
                f"This proves the rough estimate formula is fundamentally flawed."


class TestDefiniteRuleApplicationFailures:
    """Tests that should absolutely expose rule application failures"""
    
    def test_start_rule_should_prevent_pronoun_starts_but_fails(self):
        """This should definitely fail - start rule should prevent pronoun starts but doesn't"""
        chunker = CorefSafeChunker(target_size=35, max_size=70)
        
        # Create text that will naturally chunk to start with pronoun
        text = "Dr. Johnson is a renowned cardiologist at the hospital. The hospital serves many patients daily. She works there every weekday morning."
        
        chunks = chunker.chunk(text)
        
        # Look for ANY chunk that starts with a pronoun
        pronoun_starts = []
        for i, chunk in enumerate(chunks):
            sentences = chunker._split_into_sentences(chunk.text)
            if sentences:
                first_sentence = sentences[0]
                first_word = first_sentence.strip().split()[0].lower()
                
                # Check if starts with pronoun that should trigger start rule
                if first_word in {'she', 'he', 'they', 'it'}:
                    # Also check that it lacks entities (should trigger start rule)
                    if not chunker._has_named_entities(first_sentence):
                        pronoun_starts.append((i, first_sentence))
        
        # This should FAIL - start rule should prevent chunks from starting with pronouns
        assert len(pronoun_starts) == 0, \
            f"Start rule failed to prevent chunks starting with pronouns without entities: {pronoun_starts}. " \
            f"These chunks should have had antecedents prepended."
    
    def test_anaphora_hazard_rule_should_expand_but_fails(self):
        """This should definitely fail - high pronoun density should trigger expansion"""
        chunker = CorefSafeChunker(target_size=40, max_size=80)
        
        # Create sentence with very high pronoun density that should definitely trigger
        high_pronoun_sentence = "She told him it was theirs and they should keep it."
        text = f"Alice met Bob at the coffee shop yesterday. {high_pronoun_sentence} The meeting was productive."
        
        chunks = chunker.chunk(text)
        
        # Find chunk containing the high pronoun sentence
        hazard_chunk = None
        for chunk in chunks:
            if high_pronoun_sentence in chunk.text:
                hazard_chunk = chunk
                break
        
        if hazard_chunk:
            # Verify it has high pronoun density
            density = chunker._calculate_pronoun_density(high_pronoun_sentence)
            assert density >= 0.15, f"Test sentence should have high density: {density:.2%}"
            
            # The anaphora hazard rule should have expanded context to include antecedents
            # It should include "Alice met Bob" as antecedent context
            has_antecedent = "Alice met Bob" in hazard_chunk.text
            
            # This should FAIL if anaphora hazard rule isn't working
            assert has_antecedent, \
                f"Anaphora hazard rule failed to expand context. " \
                f"Sentence with {density:.2%} pronoun density should include antecedents. " \
                f"Chunk: '{hazard_chunk.text}'"
    
    def test_rules_create_chunks_exceeding_max_size(self):
        """This should definitely fail - rules can create oversized chunks"""
        chunker = CorefSafeChunker(target_size=30, max_size=50)
        
        # Create text where adding antecedents will exceed max_size
        long_antecedent = "Dr. Elizabeth Johnson conducted comprehensive research at the prestigious university laboratory."
        pronoun_sentence = "She discovered important findings."
        
        text = f"{long_antecedent} {pronoun_sentence}"
        
        # The antecedent alone might be close to max_size
        antecedent_tokens = chunker._count_tokens(long_antecedent)
        pronoun_tokens = chunker._count_tokens(pronoun_sentence)
        total_tokens = antecedent_tokens + pronoun_tokens
        
        # Ensure this would exceed max_size when combined
        assert total_tokens > chunker.max_size, \
            f"Test setup wrong - combined text should exceed max_size. " \
            f"Antecedent: {antecedent_tokens}, Pronoun: {pronoun_tokens}, " \
            f"Total: {total_tokens}, Max: {chunker.max_size}"
        
        chunks = chunker.chunk(text)
        
        # Check if any chunk exceeds max_size
        oversized_chunks = []
        for chunk in chunks:
            token_count = chunker._count_tokens(chunk.text)
            if token_count > chunker.max_size:
                oversized_chunks.append((chunk, token_count))
        
        # This should FAIL - rules should not create oversized chunks
        assert len(oversized_chunks) == 0, \
            f"Rules created {len(oversized_chunks)} chunks exceeding max_size ({chunker.max_size}): " \
            f"{[(c.text[:50] + '...', tokens) for c, tokens in oversized_chunks]}"


class TestArchitecturalIntegrationFailures:
    """Tests that expose fundamental architectural problems"""
    
    def test_sentence_lookup_fails_with_empty_results(self):
        """Force the sentence lookup to fail completely"""
        chunker = CorefSafeChunker(target_size=20, max_size=40)
        
        # Create scenario where sentence lookup will fail to find anything
        text = "A works. B works. C works. He works. D works."
        
        chunks = chunker.chunk(text) 
        sentences = chunker._split_into_sentences(text)
        
        # Manually break the lookup by providing impossible parameters
        impossible_chunk_idx = 999  # Way beyond actual chunks
        
        # This should return empty results or fail
        result = chunker._get_previous_sentences_with_entities(
            sentences, impossible_chunk_idx, chunks, max_sentences=1
        )
        
        # The result should be empty, proving the mechanism has no safeguards
        assert len(result) == 0, \
            f"Sentence lookup should return empty for impossible chunk index, but got: {result}"
        
        # Now test with real chunk but impossible sentence structure
        if len(chunks) > 1:
            # Modify chunk structure to break assumptions
            chunk = chunks[1]
            
            # Try lookup with mismatched data
            fake_sentences = ["This sentence doesn't exist in the original text."]
            result = chunker._get_previous_sentences_with_entities(
                fake_sentences, 1, chunks, max_sentences=1
            )
            
            # Should return empty or fail gracefully
            assert len(result) == 0, \
                f"Sentence lookup should fail gracefully with mismatched data, but got: {result}"