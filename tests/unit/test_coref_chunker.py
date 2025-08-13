import pytest
from flexmind.chunking import CorefSafeChunker


class TestCorefSafeChunker:
    
    @pytest.fixture
    def chunker(self):
        return CorefSafeChunker(target_size=350, max_size=600)
    
    @pytest.fixture 
    def small_chunker(self):
        """Chunker with smaller target size for easier testing"""
        return CorefSafeChunker(target_size=30, max_size=60)
    
    def test_basic_sentence_chunking(self, chunker):
        """Test that basic sentence boundaries are respected"""
        text = "John walked to the store. He bought some milk. Mary saw him there. She waved hello."
        chunks = chunker.chunk(text)
        
        # Should create chunks respecting sentence boundaries
        assert len(chunks) >= 1
        for chunk in chunks:
            # No chunk should end mid-sentence
            assert chunk.text.strip().endswith(('.', '!', '?'))
    
    def test_pronoun_density_calculation(self, chunker):
        """Test pronoun density detection"""
        # High pronoun density sentence
        high_pronoun_text = "He told her that it was theirs and they should keep it."
        density = chunker._calculate_pronoun_density(high_pronoun_text)
        assert density >= 0.15
        
        # Low pronoun density sentence  
        low_pronoun_text = "John walked to the grocery store on Main Street."
        density = chunker._calculate_pronoun_density(low_pronoun_text)
        assert density < 0.15
    
    def test_named_entity_detection(self, chunker):
        """Test named entity detection in sentences"""
        # Sentence with named entities
        entity_text = "John Smith visited New York last Tuesday."
        has_entities = chunker._has_named_entities(entity_text)
        assert has_entities
        
        # Sentence without named entities (only pronouns)
        pronoun_text = "He told her about it yesterday."
        has_entities = chunker._has_named_entities(pronoun_text)
        assert not has_entities
    
    def test_start_rule_prepending(self, small_chunker):
        """Test that chunks starting with pronouns get previous sentence prepended"""
        # Create text that will exceed target size of 100 tokens
        text = "Dr. Johnson examined the patient carefully during the morning rounds at the hospital. The examination revealed multiple concerning symptoms that required immediate attention and further investigation. He found several critical issues that needed urgent medical intervention."
        
        chunks = small_chunker.chunk(text)
        
        # Should have more than one chunk due to size
        assert len(chunks) > 1
        
        # Find the chunk that would naturally start with "He found several"
        second_chunk = None
        for chunk in chunks[1:]:  # Skip first chunk
            if "He found several" in chunk.text:
                second_chunk = chunk
                break
        
        assert second_chunk is not None, "Should find chunk starting with pronoun"
        # Due to start rule, should prepend the sentence with "Dr. Johnson"
        assert "Dr. Johnson examined" in second_chunk.text
    
    def test_anaphora_hazard_expansion(self, small_chunker):
        """Test expansion when anaphora hazard is detected"""
        # Create scenario with high pronoun density in second sentence
        text = "Alice met Bob at the downtown cafe yesterday morning for their weekly coffee meeting. The conversation lasted for quite some time as they discussed various topics. She told him that it was theirs and they should keep it with them."
        
        chunks = small_chunker.chunk(text)
        assert len(chunks) > 1, "Should create multiple chunks"
        
        # Find chunk containing the high-pronoun sentence
        hazard_chunk = None
        for chunk in chunks:
            if "She told him that it was theirs" in chunk.text:
                hazard_chunk = chunk
                break
        
        assert hazard_chunk is not None, "Should find chunk with anaphora hazard"
        
        # Due to anaphora hazard (≥15% pronoun density), should include antecedent
        assert "Alice met Bob" in hazard_chunk.text, "Should expand to include antecedent sentence"
    
    def test_paragraph_boundary_respect(self, chunker):
        """Test that paragraph boundaries can be handled without errors"""
        # Simple test: verify chunker can handle paragraph breaks without crashing
        text = "John went to the store.\n\nMary was at home."
        
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1, "Should create at least one chunk"
        
        # Verify paragraph breaks don't cause issues
        for chunk in chunks:
            assert chunk.text.strip(), "Chunks should not be empty"
            assert chunk.metadata['token_count'] > 0, "Should have token count"
    
    def test_pronoun_sentence_start_rule(self, small_chunker):
        """Test anaphora hazard bump for sentences starting with pronouns"""
        # Create text where a sentence starts with "This" 
        text = "The scientist Dr. Martinez discovered a new revolutionary compound in the laboratory after years of dedicated research and experimentation. This was groundbreaking research that changed everything we understood about chemistry."
        
        chunks = small_chunker.chunk(text)
        assert len(chunks) > 1, "Should create multiple chunks"
        
        # Find chunk that would start with "This was groundbreaking"  
        this_chunk = None
        for chunk in chunks:
            if "This was groundbreaking" in chunk.text:
                this_chunk = chunk
                break
        
        assert this_chunk is not None, "Should find chunk with demonstrative pronoun start"
        # Should include antecedent sentence due to pronoun start rule
        assert "Dr. Martinez discovered" in this_chunk.text, "Should include antecedent for pronoun sentence start"
    
    def test_overlap_preservation(self, chunker):
        """Test that overlap between chunks preserves context"""
        text = "Dr. Johnson examined the patient. He found several issues. The diagnosis was complex. She needed surgery immediately."
        chunks = chunker.chunk(text)
        
        if len(chunks) > 1:
            # Should have some overlap between adjacent chunks
            overlap_found = False
            for i in range(len(chunks) - 1):
                chunk1_sentences = chunker._split_into_sentences(chunks[i].text)
                chunk2_sentences = chunker._split_into_sentences(chunks[i + 1].text)
                
                # Check if any sentences overlap
                for sent1 in chunk1_sentences:
                    for sent2 in chunk2_sentences:
                        if sent1.strip() == sent2.strip() and sent1.strip():
                            overlap_found = True
                            break
            
            assert overlap_found
    
    def test_target_size_respect(self, chunker):
        """Test that chunks aim for target size"""
        # Create text that should result in multiple chunks
        sentences = [f"Sentence {i} has some content about topic {i}." for i in range(50)]
        text = " ".join(sentences)
        
        chunks = chunker.chunk(text)
        
        # Most chunks should be reasonably close to target size
        for chunk in chunks[:-1]:  # Exclude last chunk which might be smaller
            assert 200 <= chunker._count_tokens(chunk.text) <= 600
    
    def test_fallback_no_entities(self, chunker):
        """Test fallback when no named entities are found in window"""
        # Text with no named entities, only pronouns and common nouns
        text = "He walked to the building. It was tall and modern. They entered through the door. She looked around the lobby."
        chunks = chunker.chunk(text)
        
        # Should still create valid chunks using fallback strategy
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0
    
    def test_chunk_metadata(self, chunker):
        """Test that chunk metadata is properly set"""
        text = "John walked to the store. He bought some groceries."
        chunks = chunker.chunk(text)
        
        for i, chunk in enumerate(chunks):
            assert hasattr(chunk, 'text')
            assert hasattr(chunk, 'start_idx')
            assert hasattr(chunk, 'end_idx')
            assert hasattr(chunk, 'metadata')
            assert chunk.metadata['chunk_id'] == i
            assert 'token_count' in chunk.metadata
            assert 'sentence_count' in chunk.metadata
    
    def test_empty_and_edge_cases(self, chunker):
        """Test edge cases like empty text, single sentence, etc."""
        # Empty text
        chunks = chunker.chunk("")
        assert len(chunks) == 0
        
        # Single sentence
        chunks = chunker.chunk("John walked to the store.")
        assert len(chunks) == 1
        
        # Very long sentence
        long_sentence = "This is a very long sentence that goes on and on with many clauses and phrases and should still be handled properly by the chunker even though it exceeds the normal target size for a chunk."
        chunks = chunker.chunk(long_sentence)
        assert len(chunks) >= 1