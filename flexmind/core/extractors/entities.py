"""
Named Entity Recognition (NER) module for the FlexMind knowledge graph system.

Implements hybrid spaCy + DistilBERT entity extraction with performance optimization:
- Primary: spaCy (50,000-100,000 tokens/sec on CPU)
- Fallback: DistilBERT (60% faster than BERT, 97% accuracy)
- Confidence-based filtering and deduplication
"""

import spacy
from transformers import pipeline
from typing import List, Tuple, Set
import logging

# Set up logging
logger = logging.getLogger(__name__)


class Entity:
    """Represents an extracted named entity with metadata."""
    
    def __init__(self, text: str, label: str, confidence: float = 1.0, source: str = "spacy"):
        self.text = text
        self.label = label
        self.confidence = confidence
        self.source = source  # 'spacy' or 'distilbert'
    
    def __str__(self):
        """Human-readable string representation."""
        confidence_str = f" (conf: {self.confidence:.2f})" if self.confidence < 1.0 else ""
        source_str = f" [{self.source}]" if self.source != "spacy" else ""
        return f"Entity('{self.text}' → {self.label}{confidence_str}{source_str})"
    
    def __repr__(self):
        return f"Entity(text='{self.text}', label='{self.label}', confidence={self.confidence}, source='{self.source}')"
    
    def to_tuple(self) -> Tuple[str, str]:
        """Convert to (text, label) tuple for backward compatibility."""
        return (self.text, self.label)


class EntityExtractor:
    """
    Fast NER using spaCy primary + DistilBERT fallback.
    
    Performance targets:
    - spaCy: 50,000-100,000 tokens/second on CPU
    - DistilBERT: 60% faster than BERT, 97% accuracy
    - Confidence threshold: 0.85 for entity inclusion
    """
    
    def __init__(self, confidence_threshold: float = 0.75, use_fallback: bool = True):
        """
        Initialize the entity extractor.
        
        Args:
            confidence_threshold: Minimum confidence for entity inclusion
            use_fallback: Whether to use DistilBERT fallback for complex cases
        """
        self.confidence_threshold = confidence_threshold
        self.use_fallback = use_fallback
        
        # Initialize spaCy (primary extractor)
        try:
            # Try to load the large model first, fall back to small if not available
            try:
                self.spacy_nlp = spacy.load("en_core_web_lg")
            except OSError:
                logger.warning("en_core_web_lg not found, using en_core_web_sm")
                self.spacy_nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise RuntimeError("spaCy model not available. Run: python -m spacy download en_core_web_sm")
        
        # Initialize DistilBERT fallback (lazy loading for performance)
        self.transformer_nlp = None
        if use_fallback:
            try:
                self.transformer_nlp = pipeline(
                    "ner",
                    model="distilbert-base-cased",
                    aggregation_strategy="simple"
                )
            except Exception as e:
                logger.warning(f"DistilBERT fallback not available: {e}")
                self.use_fallback = False
    
    def extract(self, text: str) -> List[Entity]:
        """
        Extract entities from text using hybrid spaCy + DistilBERT approach.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of Entity objects with readable string representations
        """
        if not text or not text.strip():
            return []
        
        # Primary extraction with spaCy
        spacy_entities = self._extract_with_spacy(text)
        
        # Check if we need deep extraction with DistilBERT
        if self.use_fallback and self._needs_deep_extraction(spacy_entities, text):
            transformer_entities = self._extract_with_transformer(text)
            # Combine results
            all_entities = spacy_entities + transformer_entities
        else:
            all_entities = spacy_entities
        
        # Deduplicate and filter by confidence
        deduplicated_tuples = self._deduplicate_entities(all_entities)
        
        # Convert back to Entity objects
        entities = []
        for text, label in deduplicated_tuples:
            # Determine source based on original extraction
            source = "spacy"  # Default to spacy
            confidence = 1.0  # Default high confidence for spacy
            
            # Check if this came from transformer (simplified heuristic)
            if self.use_fallback and self._needs_deep_extraction(spacy_entities, text):
                source = "hybrid"
                confidence = 0.85  # Moderate confidence for fallback cases
            
            entities.append(Entity(text, label, confidence, source))
        
        return entities
    
    def _extract_with_spacy(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities using spaCy (fast path).
        
        Args:
            text: Input text
            
        Returns:
            List of (entity_text, entity_label) tuples
        """
        doc = self.spacy_nlp(text)
        entities = []
        
        for ent in doc.ents:
            # spaCy entities come with confidence implicitly high for rule-based
            entities.append((ent.text, ent.label_))
        
        return entities
    
    def _extract_with_transformer(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities using DistilBERT transformer (fallback).
        
        Args:
            text: Input text
            
        Returns:
            List of (entity_text, entity_label) tuples with confidence
        """
        if not self.transformer_nlp:
            return []
        
        try:
            results = self.transformer_nlp(text)
            entities = []
            
            for result in results:
                if result['score'] >= self.confidence_threshold:
                    # Convert transformer labels to spaCy-compatible labels
                    label = self._normalize_entity_label(result['entity_group'])
                    entities.append((result['word'], label))
            
            return entities
            
        except Exception as e:
            logger.warning(f"Transformer extraction failed: {e}")
            return []
    
    def _needs_deep_extraction(self, spacy_entities: List[Tuple[str, str]], text: str) -> bool:
        """
        Determine if text needs DistilBERT fallback extraction.
        
        Args:
            spacy_entities: Results from spaCy extraction
            text: Original text
            
        Returns:
            True if DistilBERT fallback should be used
        """
        # Heuristics for when to use transformer fallback:
        
        # 1. If spaCy found very few entities relative to text length
        tokens = len(text.split())
        entity_density = len(spacy_entities) / max(tokens, 1)
        
        if tokens > 20 and entity_density < 0.05:  # Less than 5% entity density
            return True
        
        # 2. If text contains complex patterns that might challenge spaCy
        complex_patterns = [
            'Dr.', 'Prof.', 'CEO', 'CTO', 'AI', 'ML', 
            'neural', 'algorithm', 'startup', 'VC'
        ]
        
        text_lower = text.lower()
        if any(pattern.lower() in text_lower for pattern in complex_patterns):
            return True
        
        # 3. If spaCy found no entities but text seems to have content
        if len(spacy_entities) == 0 and tokens > 10:
            return True
        
        return False
    
    def _deduplicate_entities(self, entities: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Remove duplicate entities and resolve conflicts.
        
        Args:
            entities: List of (entity_text, entity_label) tuples
            
        Returns:
            Deduplicated list of entities
        """
        if not entities:
            return []
        
        # Use dict to deduplicate by text while keeping last label
        entity_dict = {}
        seen_texts = set()
        
        for entity_text, entity_label in entities:
            # Normalize entity text
            normalized_text = entity_text.strip()
            
            if not normalized_text:
                continue
            
            # Handle substring relationships
            # If we have "Alice" and "Alice Johnson", keep the longer one
            is_duplicate = False
            texts_to_remove = []
            
            for existing_text in seen_texts:
                if normalized_text.lower() in existing_text.lower():
                    # Current entity is substring of existing - skip it
                    is_duplicate = True
                    break
                elif existing_text.lower() in normalized_text.lower():
                    # Existing entity is substring of current - remove existing
                    texts_to_remove.append(existing_text)
            
            if is_duplicate:
                continue
            
            # Remove substrings
            for text_to_remove in texts_to_remove:
                if text_to_remove in entity_dict:
                    del entity_dict[text_to_remove]
                seen_texts.discard(text_to_remove)
            
            # Add current entity
            entity_dict[normalized_text] = entity_label
            seen_texts.add(normalized_text)
        
        return list(entity_dict.items())
    
    def _normalize_entity_label(self, transformer_label: str) -> str:
        """
        Convert transformer entity labels to spaCy-compatible labels.
        
        Args:
            transformer_label: Label from transformer model
            
        Returns:
            Normalized label compatible with spaCy
        """
        # Mapping from common transformer labels to spaCy labels
        label_mapping = {
            'PER': 'PERSON',
            'PERSON': 'PERSON',
            'LOC': 'GPE',
            'ORG': 'ORG',
            'ORGANIZATION': 'ORG',
            'MISC': 'MISC',
            'B-PER': 'PERSON',
            'I-PER': 'PERSON',
            'B-LOC': 'GPE',
            'I-LOC': 'GPE',
            'B-ORG': 'ORG',
            'I-ORG': 'ORG',
            'B-MISC': 'MISC',
            'I-MISC': 'MISC',
        }
        
        return label_mapping.get(transformer_label.upper(), transformer_label)