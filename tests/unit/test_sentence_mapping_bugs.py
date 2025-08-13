"""
Tests that expose critical bugs in the sentence mapping and coreference rule implementation.
These tests are designed to FAIL with the current implementation to demonstrate the problems.
"""

import pytest
from flexmind.chunking import CorefSafeChunker


class TestSentenceMappingFailures:
    """Tests that expose sentence-to-chunk mapping failures"""
    
    @pytest.fixture
    def chunker(self):
        return CorefSafeChunker(target_size=100, max_size=200, overlap_sentences=2)
    
    def test_sentence_mapping_with_duplicate_sentences(self, chunker):
        """Test that exposes failure when identical sentences exist"""
        # Create text where the same sentence appears twice
        text = "John is a doctor. John is a doctor. He works at the hospital. Mary is a nurse. She helps patients."
        
        chunks = chunker.chunk(text)
        
        # The current implementation will find the FIRST occurrence of "John is a doctor"
        # when trying to map chunk positions, leading to incorrect lookbacks
        pronoun_chunk = None
        for chunk in chunks:
            if "He works at the hospital" in chunk.text and not chunk.text.startswith("John is a doctor"):
                pronoun_chunk = chunk
                break
        
        if pronoun_chunk:
            # This should FAIL because the sentence mapper finds the wrong "John is a doctor"
            # It should find the SECOND occurrence, not the first
            sentences = chunker._split_into_sentences(text)
            all_chunks = chunks
            
            # Try to get the chunk index for this chunk
            chunk_idx = None
            for i, c in enumerate(chunks):
                if c.text == pronoun_chunk.text:
                    chunk_idx = i
                    break
            
            if chunk_idx is not None:
                # This call should fail to find the correct position
                previous_sentences = chunker._get_previous_sentences_with_entities(
                    sentences, chunk_idx, all_chunks, max_sentences=1
                )
                
                # The bug: it will look backwards from the FIRST "John is a doctor" 
                # instead of the SECOND one, finding nothing or wrong context
                assert len(previous_sentences) > 0, \
                    "Should find previous sentences, but sentence mapping bug prevents this"
                
                # More specifically, it should find the second "John is a doctor"
                # as the immediate antecedent, but it won't due to the mapping bug
                assert "John is a doctor" in previous_sentences[0], \
                    "Should find the correct 'John is a doctor' sentence as antecedent"
    
    def test_sentence_mapping_after_rule_modifications(self, chunker):
        """Test that exposes mapping failures after chunks are modified by rules"""
        text = "Dr. Smith examined the patient carefully. The examination was thorough. He found several issues that needed attention."
        
        chunks = chunker.chunk(text)
        
        # After coreference rules are applied, chunk content changes
        # but the sentence mapping still tries to find original sentences
        modified_chunk = None
        for chunk in chunks:
            if "He found several issues" in chunk.text:
                modified_chunk = chunk
                break
        
        if modified_chunk:
            # Get the original sentences
            original_sentences = chunker._split_into_sentences(text)
            
            # The chunk has been modified by rules, so its first sentence 
            # might not match any original sentence exactly
            chunk_sentences = chunker._split_into_sentences(modified_chunk.text)
            first_chunk_sentence = chunk_sentences[0]
            
            # Try to find this sentence in the original list
            found_index = None
            for i, sent in enumerate(original_sentences):
                if sent.strip() == first_chunk_sentence.strip():
                    found_index = i
                    break
            
            # This might fail because the chunk has been modified by rules
            # and no longer matches the original sentence structure
            assert found_index is not None, \
                f"Modified chunk sentence '{first_chunk_sentence}' should be findable in original sentences, but mapping fails"
    
    def test_fallback_rough_estimate_is_wrong(self, chunker):
        """Test that exposes the 'rough estimate' fallback producing wrong results"""
        # Create text where the rough estimate (chunk_idx * 2) will be wrong
        sentences = [
            "Dr. Johnson works at the hospital.",
            "The hospital is very large.",  
            "It has many departments.",
            "The emergency room is busy.",
            "Nurses work very hard there.",
            "He treats patients every day.",  # This "He" should refer to Dr. Johnson
            "The work is challenging.",
            "She assists with operations."
        ]
        text = " ".join(sentences)
        
        chunks = chunker.chunk(text)
        
        # Find chunk containing "He treats patients"
        he_chunk = None
        chunk_idx = None
        for i, chunk in enumerate(chunks):
            if "He treats patients" in chunk.text:
                he_chunk = chunk
                chunk_idx = i
                break
        
        if he_chunk and chunk_idx:
            # Force the sentence mapping to fail by creating a scenario
            # where the sentence can't be found (simulate the bug)
            all_sentences = chunker._split_into_sentences(text)
            
            # The current implementation's fallback: chunk_idx * 2
            fallback_estimate = chunk_idx * 2
            
            # This fallback estimate is likely wrong
            if fallback_estimate < len(all_sentences):
                estimated_sentence = all_sentences[fallback_estimate]
                
                # The estimate should NOT be "He treats patients" itself
                # or any sentence that comes after it in the original sequence
                he_sentence_idx = None
                for i, sent in enumerate(all_sentences):
                    if "He treats patients" in sent:
                        he_sentence_idx = i
                        break
                
                if he_sentence_idx is not None:
                    assert fallback_estimate < he_sentence_idx, \
                        f"Fallback estimate {fallback_estimate} should be before 'He treats' at index {he_sentence_idx}, " \
                        f"but the rough estimate is wrong. Estimated sentence: '{estimated_sentence}'"


class TestCoreferenceRuleFailures:
    """Tests that expose failures in coreference rule application"""
    
    @pytest.fixture 
    def small_chunker(self):
        return CorefSafeChunker(target_size=50, max_size=100, overlap_sentences=1)
    
    def test_start_rule_fails_with_sentence_mapping_bug(self, small_chunker):
        """Test that exposes start rule failures due to sentence mapping issues"""
        # Create a scenario where start rule should work but fails due to mapping
        text = "Dr. Martinez discovered a breakthrough in AI research. The discovery was groundbreaking and changed everything. She published the results immediately. The paper became famous worldwide."
        
        chunks = small_chunker.chunk(text)
        
        # Find chunk that should trigger start rule (starts with "She published")
        she_chunk = None
        for chunk in chunks:
            if chunk.text.strip().startswith("She published"):
                she_chunk = chunk  
                break
        
        # This should FAIL - the start rule should prevent chunks from starting with "She"
        # by prepending antecedent sentences, but it fails due to sentence mapping bugs
        assert she_chunk is None, \
            "Start rule should prevent any chunk from starting with 'She published' by prepending antecedents, " \
            "but sentence mapping bug causes rule to fail"
    
    def test_anaphora_hazard_rule_gets_wrong_antecedents(self, small_chunker):
        """Test that exposes anaphora hazard rule getting wrong antecedents"""
        text = "Alice works at Google. Bob works at Microsoft. The companies are rivals. She told him that it was their responsibility to collaborate despite the competition."
        
        chunks = small_chunker.chunk(text)
        
        # Find chunk with high pronoun density sentence
        hazard_chunk = None
        for chunk in chunks:
            if "She told him that it was their responsibility" in chunk.text:
                hazard_chunk = chunk
                break
        
        if hazard_chunk:
            # The sentence has high pronoun density and should trigger anaphora hazard rule
            hazard_sentence = "She told him that it was their responsibility to collaborate despite the competition."
            density = small_chunker._calculate_pronoun_density(hazard_sentence)
            assert density >= 0.15, f"Test sentence should have high pronoun density: {density:.2%}"
            
            # Due to sentence mapping bugs, the rule might include wrong antecedents
            # It should include "Alice works at Google" (for "She") and "Bob works at Microsoft" (for "him")
            # but might get confused about which sentences to include
            
            has_alice = "Alice works at Google" in hazard_chunk.text
            has_bob = "Bob works at Microsoft" in hazard_chunk.text
            
            # This should FAIL because the anaphora hazard rule can't correctly identify
            # which previous sentences contain the right antecedents due to mapping issues
            assert has_alice and has_bob, \
                f"Anaphora hazard should include both 'Alice works at Google' and 'Bob works at Microsoft' " \
                f"as antecedents, but sentence mapping bug prevents correct antecedent selection. " \
                f"Has Alice: {has_alice}, Has Bob: {has_bob}. Chunk: '{hazard_chunk.text}'"
    
    def test_sentence_start_rule_inconsistent_application(self, small_chunker):
        """Test that exposes inconsistent application of sentence start rule"""
        # Multiple sentences starting with pronouns should all get antecedents
        text = "Dr. Kim invented a new algorithm. The algorithm was revolutionary. This changed everything in the field. It became the standard approach. That discovery won her many awards."
        
        chunks = small_chunker.chunk(text)
        
        # Find all chunks that contain sentences starting with demonstrative pronouns
        pronoun_starts = ["This changed everything", "It became the standard", "That discovery won"]
        
        for pronoun_start in pronoun_starts:
            chunk_with_pronoun = None
            for chunk in chunks:
                if pronoun_start in chunk.text:
                    chunk_with_pronoun = chunk
                    break
            
            if chunk_with_pronoun:
                # Each should include "Dr. Kim invented" as antecedent due to sentence start rule
                # But the rule application is inconsistent due to sentence mapping bugs
                has_antecedent = "Dr. Kim invented" in chunk_with_pronoun.text
                
                # This should FAIL for at least some cases due to inconsistent rule application
                assert has_antecedent, \
                    f"Sentence start rule should include 'Dr. Kim invented' as antecedent for '{pronoun_start}', " \
                    f"but rule application is inconsistent due to sentence mapping bugs. " \
                    f"Chunk: '{chunk_with_pronoun.text}'"


class TestPostProcessingArchitecturalFlaws:
    """Tests that expose architectural problems with post-processing approach"""
    
    @pytest.fixture
    def chunker(self):
        return CorefSafeChunker(target_size=80, max_size=150)
    
    def test_rules_create_oversized_chunks(self, chunker):
        """Test that exposes how coreference rules can create chunks exceeding max_size"""
        # Create text where applying all rules would exceed max_size
        long_sentences = [
            "Dr. Elizabeth Johnson led the advanced research team at the prestigious MIT AI Laboratory.",
            "The research team had been working on revolutionary neural network architectures for many months.",
            "She discovered that attention mechanisms could be significantly improved by incorporating memory structures.",
            "He colleague Dr. Michael Chen was initially skeptical about these proposed improvements to the architecture."
        ]
        text = " ".join(long_sentences)
        
        chunks = chunker.chunk(text)
        
        # Find chunks that were modified by coreference rules
        for chunk in chunks:
            token_count = chunker._count_tokens(chunk.text)
            
            # This should FAIL - rules can create chunks that exceed max_size
            # because they're applied as post-processing without size checks
            assert token_count <= chunker.max_size, \
                f"Coreference rules created chunk exceeding max_size ({chunker.max_size}). " \
                f"Chunk has {token_count} tokens: '{chunk.text}'"
    
    def test_rules_destroy_original_overlap_strategy(self, chunker):
        """Test that exposes how post-processing rules destroy the original overlap strategy"""
        text = "John works downtown. Mary works uptown. He commutes by train. She drives to work. They both work hard."
        
        # Get chunks without rules (simulate basic chunking)
        sentences = chunker._split_into_sentences(text)
        
        # Apply basic chunking first
        chunks = chunker.chunk(text)
        
        if len(chunks) > 1:
            # Check if the original overlap strategy is preserved after rule application
            original_overlap_found = False
            
            for i in range(len(chunks) - 1):
                current_sentences = chunker._split_into_sentences(chunks[i].text)
                next_sentences = chunker._split_into_sentences(chunks[i + 1].text)
                
                # Check for expected overlap from original strategy
                overlap_count = 0
                for sent1 in current_sentences[-chunker.overlap_sentences:]:
                    for sent2 in next_sentences[:chunker.overlap_sentences]:
                        if sent1.strip() == sent2.strip():
                            overlap_count += 1
                
                if overlap_count >= chunker.overlap_sentences:
                    original_overlap_found = True
                    break
            
            # This should FAIL - post-processing rules modify chunks in ways that
            # can destroy the original overlap strategy
            assert original_overlap_found, \
                "Post-processing coreference rules destroyed the original overlap strategy. " \
                "Rules should integrate with chunking, not modify chunks after the fact."