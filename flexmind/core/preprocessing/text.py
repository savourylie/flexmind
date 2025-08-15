"""
Text preprocessing module for the FlexMind knowledge graph memory system.

Handles text chunking and normalization as the foundation for all downstream
NLP processing. Implements sliding window chunking for dialogs and 
sentence-based chunking for documents.
"""

import re
from typing import List, Dict, Any


class TextChunk:
    """Represents a processed text chunk with metadata."""
    
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata
    
    def __str__(self):
        """Human-readable string representation."""
        chunk_type = self.metadata.get('chunk_type', 'unknown')
        text_type = self.metadata.get('text_type', 'unknown')
        
        preview = self.text[:100] + "..." if len(self.text) > 100 else self.text
        
        if chunk_type == 'sliding_window':
            turn_range = self.metadata.get('turn_range', (0, 0))
            return f"DialogChunk(turns={turn_range[0]}-{turn_range[1]}, text='{preview}')"
        elif chunk_type == 'sentence_based':
            token_count = self.metadata.get('token_count', 0)
            sentence_count = self.metadata.get('sentence_count', 0)
            return f"DocumentChunk(tokens={token_count}, sentences={sentence_count}, text='{preview}')"
        else:
            return f"TextChunk(type={text_type}, text='{preview}')"
    
    def __repr__(self):
        return f"TextChunk(text='{self.text[:50]}...', metadata={self.metadata})"


class TextPreprocessor:
    """
    Handles text chunking and normalization for different content types.
    
    Features:
    - Sliding window (5-10 turns) for dialog context preservation
    - Sentence-based chunking for documents with 512 token limits
    - Context overlap preservation between chunks
    - Entity-aware normalization
    """
    
    def __init__(self, window_size: int = 5, max_tokens: int = 512):
        """
        Initialize the text preprocessor.
        
        Args:
            window_size: Number of turns to include in dialog sliding window
            max_tokens: Maximum tokens per chunk for documents
        """
        self.window_size = window_size
        self.max_tokens = max_tokens
    
    def process(self, text: str, text_type: str = 'general') -> List[TextChunk]:
        """
        Process text into chunks based on content type.
        
        Args:
            text: Input text to process
            text_type: Type of text ('dialog', 'document', 'general')
        
        Returns:
            List of TextChunk objects with readable string representations
        """
        # Handle empty or whitespace-only text
        if not text or not text.strip():
            return []
        
        # Normalize the text
        normalized_text = self.normalize(text)
        
        # Choose chunking strategy based on text type
        if text_type == 'dialog':
            return self.sliding_window_chunk(normalized_text, text_type)
        elif text_type == 'document':
            return self.sentence_chunk(normalized_text, text_type)
        else:
            # Default to sentence chunking for general text
            return self.sentence_chunk(normalized_text, text_type)
    
    def sliding_window_chunk(self, text: str, text_type: str) -> List[TextChunk]:
        """
        Chunk dialog text using sliding window to preserve context.
        
        Args:
            text: Normalized dialog text
            text_type: Text type for metadata
        
        Returns:
            List of TextChunk objects with overlapping context
        """
        # Split dialog into turns (assuming speaker: message format)
        turns = []
        for line in text.split('\n'):
            line = line.strip()
            if line and ':' in line:
                turns.append(line)
        
        if len(turns) == 0:
            return []
        
        chunks = []
        
        # Create sliding windows of turns
        for i in range(0, len(turns), max(1, self.window_size - 2)):  # Overlap of 2 turns
            end_idx = min(i + self.window_size, len(turns))
            window_turns = turns[i:end_idx]
            
            chunk_text = '\n'.join(window_turns)
            metadata = {
                'text_type': text_type,
                'chunk_type': 'sliding_window',
                'turn_range': (i, end_idx - 1),
                'total_turns': len(turns)
            }
            chunk = TextChunk(chunk_text, metadata)
            chunks.append(chunk)
            
            # Stop if we've covered all turns
            if end_idx >= len(turns):
                break
        
        return chunks
    
    def sentence_chunk(self, text: str, text_type: str) -> List[TextChunk]:
        """
        Chunk document text by sentences with token limits.
        
        Args:
            text: Normalized document text
            text_type: Text type for metadata
        
        Returns:
            List of TextChunk objects respecting token limits
        """
        # Simple sentence splitting (could be enhanced with spaCy later)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            return []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            
            # If adding this sentence would exceed token limit, start new chunk
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                metadata = {
                    'text_type': text_type,
                    'chunk_type': 'sentence_based',
                    'token_count': current_tokens,
                    'sentence_count': len(current_chunk)
                }
                chunk = TextChunk(chunk_text, metadata)
                chunks.append(chunk)
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk if it has content
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            metadata = {
                'text_type': text_type,
                'chunk_type': 'sentence_based',
                'token_count': current_tokens,
                'sentence_count': len(current_chunk)
            }
            chunk = TextChunk(chunk_text, metadata)
            chunks.append(chunk)
        
        return chunks
    
    def normalize(self, text: str) -> str:
        """
        Normalize text while preserving important entities.
        
        Args:
            text: Raw input text
        
        Returns:
            Normalized text with preserved entities
        """
        # Basic normalization - preserve proper nouns and entities
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text