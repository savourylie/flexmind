"""
Unit tests for text preprocessing functionality.

Tests the core text chunking and normalization that serves as foundation
for all downstream NLP processing in the knowledge graph system.
"""

import pytest
from flexmind.core.preprocessing.text import TextPreprocessor


class TestTextPreprocessor:
    """Test text preprocessing with dialog and document chunking."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a TextPreprocessor instance for testing."""
        return TextPreprocessor()
    
    def test_dialog_sliding_window_chunking(self, preprocessor):
        """Test sliding window chunking for dialog maintains context."""
        dialog_text = """
        Alice: I just started working at OpenAI.
        Bob: That's great! When did you move to San Francisco?
        Alice: Last month. I'm working on GPT-5 now.
        Bob: How do you like the team?
        Alice: Everyone is brilliant. My manager Sarah is fantastic.
        Bob: What's your role there?
        Alice: I'm a senior research scientist focusing on alignment.
        """
        
        chunks = preprocessor.process(dialog_text, text_type='dialog')
        
        # Should create overlapping chunks with 5-turn windows
        assert len(chunks) > 0
        assert isinstance(chunks, list)
        
        # Each chunk should be a TextChunk object with text and metadata
        for chunk in chunks:
            assert hasattr(chunk, 'text')
            assert hasattr(chunk, 'metadata')
            assert chunk.metadata['text_type'] == 'dialog'
        
        # First chunk should contain early conversation
        first_chunk_text = chunks[0].text.lower()
        assert 'alice' in first_chunk_text
        assert 'openai' in first_chunk_text
    
    def test_document_sentence_chunking(self, preprocessor):
        """Test sentence-based chunking for documents with token limits."""
        document_text = """
        The knowledge graph memory system is designed for speed and efficiency.
        It uses spaCy for named entity recognition at 100,000 tokens per second.
        The system also employs DistilBERT as a fallback for complex cases.
        Neo4j serves as the graph database for storing entities and relationships.
        ChromaDB provides vector storage for unstructured content retrieval.
        """
        
        chunks = preprocessor.process(document_text, text_type='document')
        
        assert len(chunks) > 0
        
        # Each chunk should respect 512 token limit
        for chunk in chunks:
            assert len(chunk.text.split()) <= 512
            assert chunk.metadata['text_type'] == 'document'
    
    def test_normalization_preserves_entities(self, preprocessor):
        """Test that text normalization preserves important entities."""
        text_with_entities = "Alice Johnson works at OpenAI in San Francisco since 2024."
        
        chunks = preprocessor.process(text_with_entities)
        normalized_text = chunks[0].text
        
        # Should preserve proper nouns and entities
        assert 'Alice Johnson' in normalized_text
        assert 'OpenAI' in normalized_text
        assert 'San Francisco' in normalized_text
        assert '2024' in normalized_text
    
    def test_empty_text_handling(self, preprocessor):
        """Test graceful handling of empty or whitespace-only text."""
        empty_chunks = preprocessor.process("")
        whitespace_chunks = preprocessor.process("   \n\t  ")
        
        assert empty_chunks == []
        assert whitespace_chunks == []
    
    def test_chunk_overlap_preservation(self, preprocessor):
        """Test that dialog chunks preserve context overlap."""
        long_dialog = """
        Speaker1: This is turn 1.
        Speaker2: This is turn 2.
        Speaker1: This is turn 3.
        Speaker2: This is turn 4.
        Speaker1: This is turn 5.
        Speaker2: This is turn 6.
        Speaker1: This is turn 7.
        """
        
        chunks = preprocessor.process(long_dialog, text_type='dialog')
        
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            chunk1_text = chunks[0].text
            chunk2_text = chunks[1].text
            
            # Should have some shared content for context continuity
            chunk1_lines = set(chunk1_text.split('\n'))
            chunk2_lines = set(chunk2_text.split('\n'))
            overlap = chunk1_lines & chunk2_lines
            assert len(overlap) > 0, "Chunks should have overlapping context"