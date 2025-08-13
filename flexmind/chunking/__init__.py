"""
Chunking module for FlexMind

Contains coreference-safe chunking implementations that maintain 
contextual coherence for human-like memory processing.
"""

from .coref_chunker import CorefSafeChunker, Chunk

__all__ = ["CorefSafeChunker", "Chunk"]