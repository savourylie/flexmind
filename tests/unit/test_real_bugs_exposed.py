"""
Tests that expose the ACTUAL bugs by examining the internal mechanisms directly.
Based on analysis of debug output, these tests target the real issues.
"""

import pytest
from flexmind.chunking import CorefSafeChunker


class TestActualBugsExposed:
    """Tests that expose the real bugs found through debugging"""
    
    def test_sentence_mapping_finds_wrong_occurrence_of_identical_sentences(self):
        """The core bug: sentence mapping finds first occurrence, not correct one"""
        chunker = CorefSafeChunker(target_size=40, max_size=80)
        
        # Create text with identical sentences where position matters
        text = "John walked to the store. Mary saw John there. John walked to the store. He bought some groceries. The store was busy."
        
        sentences = chunker._split_into_sentences(text)
        chunks = chunker.chunk(text)
        
        # "John walked to the store" appears at positions 0 and 2
        # "He bought some groceries" is at position 3
        # The correct antecedent for "He" should be position 2, not position 0
        
        # Find chunk containing "He bought some groceries"
        he_chunk_idx = None
        for i, chunk in enumerate(chunks):
            if "He bought some groceries" in chunk.text:
                he_chunk_idx = i
                break
        
        if he_chunk_idx is not None:
            # Test the internal mapping mechanism directly
            he_chunk = chunks[he_chunk_idx]
            he_chunk_sentences = chunker._split_into_sentences(he_chunk.text)
            
            # Find the "He bought" sentence in the chunk
            he_sentence_in_chunk = None
            for sent in he_chunk_sentences:
                if "He bought some groceries" in sent:
                    he_sentence_in_chunk = sent
                    break
            
            # Now find its position in original sentence list
            found_position = None
            for i, orig_sent in enumerate(sentences):
                if orig_sent.strip() == he_sentence_in_chunk.strip():
                    found_position = i
                    break
            
            # The "He bought" should be at position 3
            assert found_position == 3, f"'He bought' should be at position 3, found at {found_position}"
            
            # Now test what _get_previous_sentences_with_entities finds
            previous = chunker._get_previous_sentences_with_entities(
                sentences, he_chunk_idx, chunks, max_sentences=1
            )
            
            if previous:
                found_previous = previous[0]
                # It should find "John walked to the store" from position 2
                # But it will likely find the one from position 0 due to the bug
                correct_previous = sentences[2]  # Position 2: "John walked to the store"
                wrong_previous = sentences[0]    # Position 0: "John walked to the store"
                
                if found_previous == wrong_previous and found_previous != correct_previous:
                    pytest.fail(f"BUG EXPOSED: Sentence mapping found wrong occurrence. " \
                              f"Found: '{found_previous}' (position 0), " \
                              f"Should find: '{correct_previous}' (position 2)")
    
    def test_start_rule_mapping_uses_wrong_chunk_position(self):
        """Test that exposes start rule using wrong chunk position due to mapping bug"""
        chunker = CorefSafeChunker(target_size=25, max_size=50)
        
        # Create text where chunk positions don't align with sentence positions
        text = "Alice is a doctor. Bob is a nurse. Carol is a patient. She needs help. Dave is a therapist."
        
        chunks = chunker.chunk(text)
        sentences = chunker._split_into_sentences(text)
        
        # Manually inspect what happens in start rule application
        for chunk_idx, chunk in enumerate(chunks):
            if chunk_idx == 0:
                continue  # Skip first chunk
                
            chunk_sentences = chunker._split_into_sentences(chunk.text)
            if not chunk_sentences:
                continue
                
            first_sentence = chunk_sentences[0]
            
            # If this triggers start rule conditions
            if (not chunker._has_named_entities(first_sentence) and 
                chunker._contains_pronouns(first_sentence)):
                
                # Test the mapping mechanism used by start rule
                found_position = None
                for i, orig_sent in enumerate(sentences):
                    if orig_sent.strip() == first_sentence.strip():
                        found_position = i
                        break
                
                # If sentence mapping fails (returns None), start rule uses fallback
                if found_position is None:
                    # This exposes the bug - sentence not found, fallback used
                    fallback_position = chunk_idx * 2
                    
                    pytest.fail(f"BUG EXPOSED: Start rule sentence mapping failed. " \
                              f"Chunk {chunk_idx} first sentence '{first_sentence}' " \
                              f"not found in original sentences. " \
                              f"Fallback position {fallback_position} will be used instead.")
    
    def test_post_processing_changes_destroy_chunk_boundaries(self):
        """Test that post-processing rule changes destroy original chunk boundaries"""
        chunker = CorefSafeChunker(target_size=35, max_size=70, overlap_sentences=1)
        
        text = "Dr. Kim invented a new algorithm. The algorithm was revolutionary. This changed everything. It became standard practice."
        
        # Get chunks and inspect the boundaries before/after rule application
        chunks = chunker.chunk(text)
        
        # Check if any chunk was modified by rules (contains sentences that weren't originally together)
        for chunk in chunks:
            chunk_sentences = chunker._split_into_sentences(chunk.text)
            
            if len(chunk_sentences) > 2:  # More than expected from normal chunking
                # Check if the sentences appear consecutively in original text
                original_sentences = chunker._split_into_sentences(text)
                
                # Find positions of chunk sentences in original text
                positions = []
                for chunk_sent in chunk_sentences:
                    for i, orig_sent in enumerate(original_sentences):
                        if orig_sent.strip() == chunk_sent.strip():
                            if i not in positions:  # Avoid duplicates
                                positions.append(i)
                            break
                
                # Check if positions are consecutive (allowing for overlap_sentences)
                positions.sort()
                
                # If there are gaps > overlap_sentences, rules modified the chunk
                for i in range(len(positions) - 1):
                    gap = positions[i + 1] - positions[i]
                    if gap > chunker.overlap_sentences:
                        pytest.fail(f"BUG EXPOSED: Post-processing rules created non-consecutive chunk. " \
                                  f"Chunk contains sentences from positions {positions}, " \
                                  f"with gap of {gap} > overlap_sentences ({chunker.overlap_sentences}). " \
                                  f"Chunk: '{chunk.text}'")
    
    def test_rules_dont_handle_size_constraints(self):
        """Test that exposes rules not checking size constraints when adding content"""
        # Use very small max_size to force the issue
        chunker = CorefSafeChunker(target_size=15, max_size=25)
        
        # Create text where rule addition would exceed max_size
        text = "Dr. Johnson works at the big hospital downtown. She helps patients every single day."
        
        chunks = chunker.chunk(text)
        
        # Check if any chunk exceeds max_size due to rule modifications
        oversized_chunks = []
        for chunk in chunks:
            token_count = chunker._count_tokens(chunk.text)
            if token_count > chunker.max_size:
                oversized_chunks.append((chunk, token_count))
        
        if oversized_chunks:
            pytest.fail(f"BUG EXPOSED: Rules created {len(oversized_chunks)} chunks exceeding max_size. " \
                      f"Max allowed: {chunker.max_size}, " \
                      f"Oversized chunks: {[(chunk.text[:30] + '...', tokens) for chunk, tokens in oversized_chunks]}")
    
    def test_duplicate_content_from_overlapping_rule_applications(self):
        """Test that exposes duplicate content when multiple rules trigger on same chunk"""
        chunker = CorefSafeChunker(target_size=30, max_size=60, overlap_sentences=1)
        
        # Create scenario where multiple rules might add the same antecedent
        text = "Alice works at Google. Bob works at Microsoft. She told him that it was their responsibility to collaborate."
        
        chunks = chunker.chunk(text)
        
        # Find chunk that might trigger multiple rules
        multi_rule_chunk = None
        for chunk in chunks:
            if "She told him that it was their" in chunk.text:
                multi_rule_chunk = chunk
                break
        
        if multi_rule_chunk:
            # Check if the same sentence appears multiple times due to different rules
            chunk_sentences = chunker._split_into_sentences(multi_rule_chunk.text)
            sentence_counts = {}
            
            for sent in chunk_sentences:
                sent_normalized = sent.strip()
                sentence_counts[sent_normalized] = sentence_counts.get(sent_normalized, 0) + 1
            
            # Look for duplicates
            duplicates = {sent: count for sent, count in sentence_counts.items() if count > 1}
            
            if duplicates:
                pytest.fail(f"BUG EXPOSED: Multiple rules created duplicate sentences in chunk. " \
                          f"Duplicates: {duplicates}. " \
                          f"Chunk: '{multi_rule_chunk.text}'")
    
    def test_fallback_rough_estimate_beyond_bounds(self):
        """Test that exposes fallback producing out-of-bounds array access"""
        chunker = CorefSafeChunker(target_size=15, max_size=30)
        
        # Create very short text where rough estimate will exceed bounds
        text = "Alice works here. Bob helps Alice. She is busy."  # 3 clear sentences
        
        sentences = chunker._split_into_sentences(text)
        chunks = chunker.chunk(text)
        
        # Should have 3 sentences: ["Alice works here.", "Bob helps Alice.", "She is busy."]
        assert len(sentences) == 3, f"Expected 3 sentences, got {len(sentences)}: {sentences}"
        
        # If we have multiple chunks and the last one uses fallback
        if len(chunks) > 1:
            last_chunk_idx = len(chunks) - 1
            
            # The rough estimate would be: last_chunk_idx * 2
            rough_estimate = last_chunk_idx * 2
            
            # This could easily exceed the sentence count (3)
            if rough_estimate >= len(sentences):
                pytest.fail(f"BUG EXPOSED: Rough estimate fallback produces out-of-bounds index. " \
                          f"Chunk index: {last_chunk_idx}, " \
                          f"Rough estimate: {rough_estimate}, " \
                          f"Sentence count: {len(sentences)} (max index: {len(sentences)-1}). " \
                          f"This would cause array bounds error.")


class TestInternalStateInconsistencies:
    """Tests that expose inconsistencies in internal state tracking"""
    
    def test_chunk_metadata_inconsistent_with_actual_content(self):
        """Test that chunk metadata doesn't match actual chunk content after rule modifications"""
        chunker = CorefSafeChunker(target_size=40, max_size=80)
        
        text = "Dr. Martinez led the team. The research was complex. She discovered new methods. The results were groundbreaking."
        
        chunks = chunker.chunk(text)
        
        for chunk in chunks:
            # Check if metadata matches actual content
            actual_tokens = chunker._count_tokens(chunk.text)
            actual_sentences = len(chunker._split_into_sentences(chunk.text))
            actual_has_entities = chunker._has_named_entities(chunk.text)
            actual_pronoun_density = chunker._calculate_pronoun_density(chunk.text)
            
            metadata = chunk.metadata
            
            # These should match, but might not after rule modifications
            inconsistencies = []
            
            if metadata['token_count'] != actual_tokens:
                inconsistencies.append(f"token_count: metadata={metadata['token_count']}, actual={actual_tokens}")
            
            if metadata['sentence_count'] != actual_sentences:
                inconsistencies.append(f"sentence_count: metadata={metadata['sentence_count']}, actual={actual_sentences}")
            
            if metadata['has_entities'] != actual_has_entities:
                inconsistencies.append(f"has_entities: metadata={metadata['has_entities']}, actual={actual_has_entities}")
            
            # Pronoun density might change significantly after rule modifications
            density_diff = abs(metadata['pronoun_density'] - actual_pronoun_density)
            if density_diff > 0.05:  # 5% tolerance
                inconsistencies.append(f"pronoun_density: metadata={metadata['pronoun_density']:.2%}, actual={actual_pronoun_density:.2%}")
            
            if inconsistencies:
                pytest.fail(f"BUG EXPOSED: Chunk metadata inconsistent with actual content after rule modifications. " \
                          f"Inconsistencies: {inconsistencies}. " \
                          f"Chunk: '{chunk.text[:50]}...'")
    
    def test_start_idx_end_idx_wrong_after_modifications(self):
        """Test that start_idx and end_idx are wrong after chunk modifications"""
        chunker = CorefSafeChunker(target_size=35, max_size=70)
        
        text = "Alice works downtown. Mary works uptown. He commutes daily. She drives to work."
        
        chunks = chunker.chunk(text)
        
        for chunk in chunks:
            # Check if start_idx and end_idx make sense
            calculated_length = len(chunk.text)
            metadata_length = chunk.end_idx - chunk.start_idx
            
            # These might not match if rules modified the chunk content
            if abs(calculated_length - metadata_length) > 5:  # Small tolerance for whitespace
                pytest.fail(f"BUG EXPOSED: Chunk start_idx/end_idx inconsistent with actual text length. " \
                          f"Text length: {calculated_length}, " \
                          f"Metadata length: {metadata_length} (start={chunk.start_idx}, end={chunk.end_idx}). " \
                          f"Chunk: '{chunk.text}'")
            
            # Also check if the indices make sense relative to original text
            try:
                extracted_text = text[chunk.start_idx:chunk.end_idx]
                
                # After rule modifications, this extraction might not match chunk.text
                if extracted_text != chunk.text:
                    # This is expected to fail due to rule modifications
                    pytest.fail(f"BUG EXPOSED: Chunk indices don't correspond to actual chunk text. " \
                              f"Extracted from original text[{chunk.start_idx}:{chunk.end_idx}]: '{extracted_text}' " \
                              f"Actual chunk text: '{chunk.text}' " \
                              f"This proves rule modifications break index tracking.")
            except Exception:
                # Index out of bounds or other error also proves the bug
                pass