import pytest
from flexmind.chunking import CorefSafeChunker


class TestCorefSpecificBehaviors:
    """Tests that specifically verify coreference-safe behaviors beyond basic overlap"""
    
    @pytest.fixture 
    def micro_chunker(self):
        """Very small chunker to force specific behaviors"""
        return CorefSafeChunker(target_size=15, max_size=25, overlap_sentences=1)
    
    def test_start_rule_adds_context_beyond_overlap(self, micro_chunker):
        """Test that start rule adds MORE context than just normal overlap"""
        # Create text where normal overlap would only include 1 sentence,
        # but start rule should add MORE when chunk starts with pronoun + no entities
        text = "Dr. Johnson works at the hospital. The building is very tall. He treats many patients there."
        
        chunks = micro_chunker.chunk(text)
        assert len(chunks) >= 2, "Should create multiple chunks"
        
        # Find chunk that contains "He treats" - this will contain start rule expansion
        pronoun_chunk = None
        for chunk in chunks:
            if "He treats many patients" in chunk.text:
                pronoun_chunk = chunk
                break
        
        assert pronoun_chunk is not None, "Should find chunk containing pronoun sentence"
        
        # Count sentences in the chunk - start rule should add extra context
        sentences = micro_chunker._split_into_sentences(pronoun_chunk.text)
        # Normal overlap = 1 sentence + current = 2 sentences
        # Start rule should add more context = 3+ sentences (includes "Dr. Johnson" sentence)
        assert len(sentences) >= 3, f"Start rule should add extra context beyond normal overlap. Got {len(sentences)} sentences: {sentences}"
        
        # Verify the start rule specifically added the antecedent
        assert "Dr. Johnson works at the hospital" in pronoun_chunk.text, "Start rule should include Dr. Johnson sentence as antecedent"
    
    def test_anaphora_hazard_detection_triggers(self, micro_chunker):
        """Test that anaphora hazard specifically triggers context expansion"""
        # Create sentence with exactly 15%+ pronoun density to trigger hazard detection
        text = "Alice met Bob yesterday. She told him it was theirs."  # ~17% pronoun density
        
        chunks = micro_chunker.chunk(text)
        
        # Find chunk with the high-pronoun sentence
        hazard_chunk = None  
        for chunk in chunks:
            if "She told him it was theirs" in chunk.text:
                hazard_chunk = chunk
                break
                
        assert hazard_chunk is not None, "Should find chunk with anaphora hazard"
        
        # Verify that anaphora hazard was detected
        hazard_sentence = "She told him it was theirs."
        density = micro_chunker._calculate_pronoun_density(hazard_sentence)
        assert density >= 0.15, f"Test sentence should have ≥15% pronoun density, got {density:.2%}"
        
        # This should FAIL - anaphora hazard detection should add extra context
        # beyond what normal overlap would provide
        assert "Alice met Bob" in hazard_chunk.text, "Anaphora hazard should expand context to include antecedents"
    
    def test_sentence_start_pronoun_rule(self, micro_chunker):
        """Test sentences starting with specific pronouns get extra antecedent"""  
        text = "Dr. Martinez discovered something. This was very important research."
        
        chunks = micro_chunker.chunk(text)
        
        # Find chunk containing sentence that starts with "This"
        this_chunk = None
        for chunk in chunks:
            if "This was very important" in chunk.text:
                this_chunk = chunk
                break
                
        assert this_chunk is not None, "Should find chunk with 'This' start"
        
        # Check if sentence starts with demonstrative pronoun
        sentences = micro_chunker._split_into_sentences(this_chunk.text)
        has_this_start = any(sent.strip().startswith("This") for sent in sentences)
        assert has_this_start, "Should contain sentence starting with 'This'"
        
        # This should FAIL - sentence start rule should ensure antecedent is included
        assert "Dr. Martinez discovered" in this_chunk.text, "Sentence starting with 'This' should include antecedent"
    
    def test_no_entities_fallback_strategy(self, micro_chunker):
        """Test fallback when no named entities are found in reasonable window"""
        # Text with no named entities, only pronouns and common nouns  
        text = "The person walked to the building. He entered through the door. She was in the lobby."
        
        chunks = micro_chunker.chunk(text)
        
        # Should still create valid chunks even without named entities
        assert len(chunks) >= 1, "Should create chunks even without named entities"
        
        # Find chunk that might start with pronoun but has no named entities
        no_entity_chunk = None
        for chunk in chunks:
            if ("He entered" in chunk.text or "She was" in chunk.text):
                no_entity_chunk = chunk
                break
        
        if no_entity_chunk:
            # This should FAIL - fallback should include concrete nouns as anchors
            # when no named entities are available
            sentences = micro_chunker._split_into_sentences(no_entity_chunk.text)
            assert len(sentences) >= 2, "Fallback should include concrete noun context"
            assert any("person" in sent or "building" in sent for sent in sentences), \
                "Fallback should anchor with concrete nouns when no named entities exist"