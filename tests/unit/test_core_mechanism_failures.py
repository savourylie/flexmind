"""
Tests that directly verify the core mechanisms and expose their failures.
These tests check the fundamental assumptions that the coreference rules depend on.
"""

import pytest
from flexmind.chunking import CorefSafeChunker


class TestSentenceLookupMechanism:
    """Tests that verify the sentence lookup mechanism works correctly"""
    
    @pytest.fixture
    def chunker(self):
        return CorefSafeChunker(target_size=60, max_size=120)
    
    def test_sentence_position_lookup_accuracy(self, chunker):
        """Test that _get_previous_sentences_with_entities finds correct sentence positions"""
        # Create text with clear sentence structure
        sentences = [
            "Alice is a scientist.",           # 0
            "Bob is an engineer.",            # 1  
            "Carol works with data.",         # 2
            "She analyzes complex patterns.", # 3 - should look back to find "Carol"
            "Dave helps with coding.",        # 4
            "He writes efficient algorithms." # 5 - should look back to find "Dave"
        ]
        text = " ".join(sentences)
        
        chunks = chunker.chunk(text)
        all_sentences = chunker._split_into_sentences(text)
        
        # Verify we have the expected sentences
        assert len(all_sentences) == 6, f"Should have 6 sentences, got {len(all_sentences)}"
        
        # Test the lookup mechanism directly for each chunk
        for chunk_idx, chunk in enumerate(chunks):
            chunk_sentences = chunker._split_into_sentences(chunk.text)
            if not chunk_sentences:
                continue
                
            first_sentence = chunk_sentences[0]
            
            # Find the actual position of this sentence in the original list
            actual_position = None
            for i, orig_sent in enumerate(all_sentences):
                if orig_sent.strip() == first_sentence.strip():
                    actual_position = i
                    break
            
            if actual_position is None:
                # This is a major bug - chunk sentence not found in original
                pytest.fail(f"Chunk {chunk_idx} first sentence '{first_sentence}' not found in original sentences")
            
            # Now test if _get_previous_sentences_with_entities gets the same position
            previous_sentences = chunker._get_previous_sentences_with_entities(
                all_sentences, chunk_idx, chunks, max_sentences=1
            )
            
            if chunk_idx > 0 and actual_position > 0:
                # Should find the sentence immediately before actual_position
                expected_previous = all_sentences[actual_position - 1]
                
                if previous_sentences:
                    found_previous = previous_sentences[0]
                    
                    # This should FAIL - the mechanism doesn't find the correct previous sentence
                    assert found_previous == expected_previous, \
                        f"Sentence lookup failed for chunk {chunk_idx}. " \
                        f"Expected previous: '{expected_previous}' (position {actual_position-1}), " \
                        f"Got: '{found_previous}'. " \
                        f"Chunk starts with: '{first_sentence}' (position {actual_position})"
    
    def test_multiple_identical_sentences_confusion(self, chunker):
        """Test that multiple identical sentences confuse the lookup mechanism"""
        text = "John walked to the store. John walked to the store. He bought some milk. John walked to the store. She saw him there."
        
        sentences = chunker._split_into_sentences(text)
        chunks = chunker.chunk(text)
        
        # The sentence "John walked to the store" appears 3 times at positions 0, 1, and 3
        identical_sentence = "John walked to the store."
        positions = []
        for i, sent in enumerate(sentences):
            if sent.strip() == identical_sentence.strip():
                positions.append(i)
        
        assert len(positions) >= 3, f"Should find at least 3 identical sentences, got {positions}"
        
        # Find chunk containing "He bought some milk"
        he_chunk_idx = None
        for i, chunk in enumerate(chunks):
            if "He bought some milk" in chunk.text:
                he_chunk_idx = i
                break
        
        if he_chunk_idx is not None:
            # The "He bought" sentence should be at position 2 (after the second "John walked")
            # But the lookup mechanism will find the FIRST "John walked" at position 0
            previous_sentences = chunker._get_previous_sentences_with_entities(
                sentences, he_chunk_idx, chunks, max_sentences=1
            )
            
            if previous_sentences:
                # This should FAIL - it finds the wrong identical sentence
                found_sentence = previous_sentences[0]
                
                # It should find the sentence immediately before "He bought" (position 1)
                # which is the second "John walked to the store"
                # But it will likely find the first one (position 0)
                he_position = None
                for i, sent in enumerate(sentences):
                    if "He bought some milk" in sent:
                        he_position = i
                        break
                
                if he_position and he_position > 0:
                    correct_previous = sentences[he_position - 1]
                    
                    # This will fail because it finds position 0 instead of position 1
                    assert found_sentence == correct_previous, \
                        f"Multiple identical sentences confuse lookup. " \
                        f"'He bought' is at position {he_position}, " \
                        f"should find previous at position {he_position-1}: '{correct_previous}', " \
                        f"but got: '{found_sentence}' (likely from position 0)"


class TestRoughEstimateFallback:
    """Tests that expose the problems with the rough estimate fallback"""
    
    @pytest.fixture
    def chunker(self):
        return CorefSafeChunker(target_size=40, max_size=80)
    
    def test_rough_estimate_formula_is_wrong(self, chunker):
        """Test that the rough estimate formula (chunk_idx * 2) produces wrong results"""
        # Create text where sentences don't align with chunk_idx * 2
        sentences = [
            "Sentence zero has content.",     # 0
            "Sentence one has content.",     # 1  
            "Sentence two has content.",     # 2
            "Sentence three has content.",   # 3
            "Sentence four has content.",    # 4
            "Sentence five has content.",    # 5
            "He refers to someone earlier.", # 6 - pronoun that triggers fallback
            "Sentence seven has content."    # 7
        ]
        text = " ".join(sentences)
        
        chunks = chunker.chunk(text)
        all_sentences = chunker._split_into_sentences(text)
        
        # Find chunk containing the pronoun
        he_chunk_idx = None
        for i, chunk in enumerate(chunks):
            if "He refers to someone" in chunk.text:
                he_chunk_idx = i
                break
        
        if he_chunk_idx is not None and he_chunk_idx > 0:
            # The rough estimate would be: he_chunk_idx * 2
            rough_estimate = he_chunk_idx * 2
            
            # Find the actual position of "He refers to someone"
            actual_he_position = None
            for i, sent in enumerate(all_sentences):
                if "He refers to someone" in sent:
                    actual_he_position = i
                    break
            
            if actual_he_position is not None:
                # This should FAIL - the rough estimate is usually wrong
                assert rough_estimate < actual_he_position, \
                    f"Rough estimate formula is wrong. " \
                    f"Chunk {he_chunk_idx} * 2 = {rough_estimate}, " \
                    f"but 'He refers' is actually at position {actual_he_position}. " \
                    f"The estimate should be less than the actual position to look backwards correctly."
                
                # Even worse, the estimate might be >= actual position, making lookback impossible
                if rough_estimate >= actual_he_position:
                    pytest.fail(f"Rough estimate {rough_estimate} >= actual position {actual_he_position} - " \
                              f"cannot look backwards from this position!")


class TestRuleApplicationMechanism:
    """Tests that verify individual rule application mechanisms"""
    
    @pytest.fixture
    def chunker(self):
        return CorefSafeChunker(target_size=50, max_size=100)
    
    def test_start_rule_detection_accuracy(self, chunker):
        """Test that start rule correctly detects when it should trigger"""
        # Create chunks where some should trigger start rule and others shouldn't
        test_cases = [
            ("She walked to the store.", True),   # Should trigger: pronoun + no entities
            ("Dr. Smith walked to the store.", False),  # Should NOT trigger: has entity
            ("The building was tall.", False),    # Should NOT trigger: no pronouns  
            ("He and Dr. Jones met.", False),     # Should NOT trigger: has entity despite pronoun
            ("It was very cold outside.", True),  # Should trigger: pronoun + no entities
        ]
        
        for sentence, should_trigger in test_cases:
            has_entities = chunker._has_named_entities(sentence)
            has_pronouns = chunker._contains_pronouns(sentence)
            
            expected_trigger = not has_entities and has_pronouns
            
            # This should FAIL if the detection logic is wrong
            assert expected_trigger == should_trigger, \
                f"Start rule detection wrong for '{sentence}'. " \
                f"Has entities: {has_entities}, Has pronouns: {has_pronouns}, " \
                f"Expected trigger: {should_trigger}, Actual: {expected_trigger}"
    
    def test_anaphora_hazard_threshold_accuracy(self, chunker):
        """Test that anaphora hazard detection uses correct threshold (15%)"""
        test_cases = [
            ("She told him it was theirs.", True),    # High pronoun density
            ("John walked to the store.", False),     # No pronouns
            ("The building is very tall.", False),    # No pronouns
            ("He said it.", True),                    # High density (2/3 = 66%)
            ("He walked to the building.", True),     # 16.67% density (1/6 tokens) > 15% threshold
        ]
        
        for sentence, should_be_hazard in test_cases:
            density = chunker._calculate_pronoun_density(sentence)
            is_hazard = density >= 0.15
            
            # This might FAIL if threshold calculation is wrong
            assert is_hazard == should_be_hazard, \
                f"Anaphora hazard detection wrong for '{sentence}'. " \
                f"Density: {density:.2%}, Expected hazard: {should_be_hazard}, " \
                f"Actual hazard: {is_hazard} (>= 15%: {density >= 0.15})"
    
    def test_sentence_start_pronoun_detection(self, chunker):
        """Test that sentence start pronoun rule detects correct pronouns"""
        hazard_pronouns = {'he', 'she', 'they', 'this', 'that', 'it'}
        
        test_cases = [
            ("He walked to the store.", True),
            ("She found the answer.", True), 
            ("They worked together.", True),
            ("This was important.", True),
            ("That seemed wrong.", True),
            ("It was very cold.", True),
            ("John walked to the store.", False),  # Doesn't start with pronoun
            ("The building was tall.", False),     # Doesn't start with pronoun
            ("Yesterday he walked.", False),       # Pronoun not at start
        ]
        
        for sentence, should_trigger in test_cases:
            first_word = sentence.strip().split()[0].lower() if sentence.strip() else ""
            triggers = first_word in hazard_pronouns
            
            # This should FAIL if pronoun detection is wrong
            assert triggers == should_trigger, \
                f"Sentence start pronoun detection wrong for '{sentence}'. " \
                f"First word: '{first_word}', Expected trigger: {should_trigger}, " \
                f"Actual trigger: {triggers}, Hazard pronouns: {hazard_pronouns}"


class TestIntegrationMechanismFlaws:
    """Tests that expose how the rules don't integrate properly with chunking"""
    
    @pytest.fixture 
    def chunker(self):
        return CorefSafeChunker(target_size=70, max_size=140, overlap_sentences=2)
    
    def test_rules_applied_after_overlap_breaks_assumptions(self, chunker):
        """Test that shows rules assume original text but chunks already have overlap"""
        text = "Dr. Anderson led the team. The team worked on AI. She made discoveries. He colleague Dr. Brown was impressed."
        
        # Get the chunks after basic chunking with overlap
        chunks = chunker.chunk(text)
        original_sentences = chunker._split_into_sentences(text)
        
        # After overlap, chunks might contain:
        # Chunk 1: "Dr. Anderson led the team. The team worked on AI."
        # Chunk 2: "The team worked on AI. She made discoveries."  (with overlap)
        # Chunk 3: "She made discoveries. He colleague Dr. Brown was impressed." (with overlap)
        
        for i, chunk in enumerate(chunks):
            chunk_sentences = chunker._split_into_sentences(chunk.text)
            
            if len(chunk_sentences) > 1:
                # If this chunk has overlap sentences, the rule application
                # will be confused about which sentences are "original" vs "overlap"
                
                # Try to find where this chunk "really" starts in the original sequence
                first_sentence = chunk_sentences[0]
                
                # Count how many times this sentence appears in the chunk
                sentence_count_in_chunk = sum(1 for s in chunk_sentences if s.strip() == first_sentence.strip())
                
                if sentence_count_in_chunk > 1:
                    # This should FAIL - overlapped sentences break rule assumptions
                    pytest.fail(f"Chunk {i} has overlapped sentence '{first_sentence}' " \
                              f"appearing {sentence_count_in_chunk} times. " \
                              f"Rules cannot determine original vs overlap context.")
    
    def test_post_processing_destroys_chunk_size_constraints(self, chunker):
        """Test that post-processing rules can violate size constraints"""
        # Create scenario where adding antecedents exceeds max_size
        long_antecedent = "Dr. Elizabeth Johnson conducted groundbreaking research at the prestigious MIT Artificial Intelligence Laboratory with her dedicated team of brilliant researchers."
        pronoun_sentence = "She discovered revolutionary techniques."
        
        text = f"{long_antecedent} {pronoun_sentence}"
        
        chunks = chunker.chunk(text)
        
        for chunk in chunks:
            token_count = chunker._count_tokens(chunk.text) 
            
            # This should FAIL - post-processing can exceed max_size
            assert token_count <= chunker.max_size, \
                f"Post-processing created oversized chunk: {token_count} > {chunker.max_size} tokens. " \
                f"Rules don't respect size constraints: '{chunk.text}'"