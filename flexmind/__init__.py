"""
FlexMind: Human-like Memory Mechanism

A memory system inspired by "Why We Remember" that prioritizes flexibility 
and contextual adaptation over static, photographic accuracy.
"""

from .chunking.coref_chunker import CorefSafeChunker

__version__ = "0.1.0"
__author__ = "FlexMind Team"

__all__ = ["CorefSafeChunker"]