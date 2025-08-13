import spacy
from dataclasses import dataclass
from typing import List, Dict, Any
import re


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any]


class CorefSafeChunker:
    """
    Coreference-safe text chunker that maintains context for pronouns and anaphora.
    
    Features:
    - Respects sentence/paragraph boundaries
    - Maintains overlap for antecedent preservation
    - Starts chunks with grounding content (named entities)
    - Expands context when anaphora risk is detected
    """
    
    PRONOUNS = {
        'personal': ['he', 'she', 'it', 'they', 'him', 'her', 'them'],
        'possessive': ['his', 'her', 'its', 'their', 'hers', 'theirs'],
        'demonstrative': ['this', 'that', 'these', 'those'],
        'reflexive': ['himself', 'herself', 'itself', 'themselves']
    }
    
    def __init__(self, target_size: int = 350, max_size: int = 600, overlap_sentences: int = 2):
        """
        Initialize the chunker.
        
        Args:
            target_size: Target number of tokens per chunk
            max_size: Maximum tokens per chunk (hard limit)
            overlap_sentences: Number of sentences to overlap between chunks
        """
        self.target_size = target_size
        self.max_size = max_size
        self.overlap_sentences = overlap_sentences
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    
    def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk text using coreference-safe strategy.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []
        
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return [self._create_chunk(sentences[0], 0)]
        
        chunks = []
        current_sentences = []
        current_token_count = 0
        sentence_start_idx = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self._count_tokens(sentence)
            
            # Check if adding this sentence would exceed target size
            if current_token_count + sentence_tokens > self.target_size and current_sentences:
                # Create chunk from current sentences
                chunk_text = " ".join(current_sentences)
                chunk = self._create_chunk(chunk_text, sentence_start_idx, len(chunks))
                chunks.append(chunk)
                
                # Prepare next chunk with overlap
                overlap_sentences = current_sentences[-self.overlap_sentences:] if len(current_sentences) >= self.overlap_sentences else current_sentences[:]
                current_sentences = overlap_sentences + [sentence]
                current_token_count = sum(self._count_tokens(s) for s in current_sentences)
                sentence_start_idx = i - len(overlap_sentences)
            else:
                current_sentences.append(sentence)
                current_token_count += sentence_tokens
        
        # Add final chunk if there are remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk = self._create_chunk(chunk_text, sentence_start_idx, len(chunks))
            chunks.append(chunk)
        
        # Apply coreference-safe post-processing
        chunks = self._apply_coref_rules(chunks, sentences)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    def _detect_paragraph_boundaries(self, text: str) -> List[int]:
        """Detect paragraph boundaries (double newlines) and return sentence indices."""
        sentences = self._split_into_sentences(text)
        boundaries = []
        
        # Reconstruct the original text positions
        current_pos = 0
        for i, sentence in enumerate(sentences):
            # Find the sentence in the original text
            sent_start = text.find(sentence, current_pos)
            if sent_start == -1:
                continue
            
            sent_end = sent_start + len(sentence)
            current_pos = sent_end
            
            # Check if there's a paragraph break after this sentence
            remaining_text = text[sent_end:].lstrip()
            if remaining_text.startswith('\n'):
                boundaries.append(i)
        
        return boundaries
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using spaCy tokenizer."""
        doc = self.nlp(text)
        return len(doc)
    
    def _calculate_pronoun_density(self, text: str) -> float:
        """Calculate the density of pronouns in text."""
        doc = self.nlp(text)
        if len(doc) == 0:
            return 0.0
        
        pronoun_count = 0
        all_pronouns = set()
        for pronoun_type in self.PRONOUNS.values():
            all_pronouns.update(pronoun_type)
        
        for token in doc:
            if token.lemma_.lower() in all_pronouns or token.pos_ == "PRON":
                pronoun_count += 1
        
        return pronoun_count / len(doc)
    
    def _has_named_entities(self, text: str) -> bool:
        """Check if text contains substantial named entities (not just dates/times)."""
        doc = self.nlp(text)
        # Filter out temporal entities that don't provide good grounding
        substantial_entities = [ent for ent in doc.ents 
                               if ent.label_ not in {'DATE', 'TIME', 'CARDINAL', 'ORDINAL'}]
        return len(substantial_entities) > 0
    
    def _create_chunk(self, text: str, start_idx: int, chunk_id: int = 0) -> Chunk:
        """Create a Chunk object with metadata."""
        sentences = self._split_into_sentences(text)
        
        metadata = {
            'chunk_id': chunk_id,
            'token_count': self._count_tokens(text),
            'sentence_count': len(sentences),
            'has_entities': self._has_named_entities(text),
            'pronoun_density': self._calculate_pronoun_density(text)
        }
        
        return Chunk(
            text=text,
            start_idx=start_idx,
            end_idx=start_idx + len(text),
            metadata=metadata
        )
    
    def _apply_coref_rules(self, chunks: List[Chunk], sentences: List[str]) -> List[Chunk]:
        """Apply coreference-safe rules to chunks."""
        if len(chunks) <= 1:
            return chunks
        
        updated_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_sentences = self._split_into_sentences(chunk.text)
            if not chunk_sentences:
                updated_chunks.append(chunk)
                continue
            
            # Apply start rule: check if chunk starts with pronouns but lacks entities
            modified_chunk = self._apply_start_rule(chunk, chunks, sentences, i)
            
            # Apply anaphora hazard expansion
            modified_chunk = self._apply_anaphora_hazard_rule(modified_chunk, chunks, sentences, i)
            
            # Apply sentence start pronoun rule
            modified_chunk = self._apply_sentence_start_rule(modified_chunk, chunks, sentences, i)
            
            updated_chunks.append(modified_chunk)
        
        return updated_chunks
    
    def _apply_start_rule(self, chunk: Chunk, all_chunks: List[Chunk], all_sentences: List[str], chunk_idx: int) -> Chunk:
        """Apply start rule: prepend previous sentence if chunk starts with pronouns and lacks entities."""
        if chunk_idx == 0:  # First chunk, no previous context
            return chunk
        
        chunk_sentences = self._split_into_sentences(chunk.text)
        if not chunk_sentences:
            return chunk
        
        # Check if FIRST sentence lacks named entities and contains pronouns
        first_sentence = chunk_sentences[0]
        
        needs_prepending = (not self._has_named_entities(first_sentence) and 
                          self._contains_pronouns(first_sentence))
        
        if not needs_prepending:
            return chunk
        
        # Find the most recent sentence with named entities to prepend
        previous_sentences = self._get_previous_sentences_with_entities(all_sentences, chunk_idx, all_chunks)
        
        if previous_sentences:
            # Avoid duplication - only add sentences not already in chunk
            unique_previous = [sent for sent in previous_sentences if sent not in chunk_sentences]
            if unique_previous:
                # Prepend the antecedent sentence(s)
                new_text = " ".join(unique_previous + chunk_sentences)
                return self._create_chunk(new_text, chunk.start_idx, chunk.metadata['chunk_id'])
        
        return chunk
    
    def _apply_anaphora_hazard_rule(self, chunk: Chunk, all_chunks: List[Chunk], all_sentences: List[str], chunk_idx: int) -> Chunk:
        """Apply anaphora hazard rule: expand context for high pronoun density sentences."""
        chunk_sentences = self._split_into_sentences(chunk.text)
        if not chunk_sentences:
            return chunk
        
        # Check each sentence for anaphora hazard (≥15% pronoun density)
        has_hazard = False
        for sentence in chunk_sentences:
            if self._calculate_pronoun_density(sentence) >= 0.15:
                has_hazard = True
                break
        
        if not has_hazard:
            return chunk
        
        # Include extra antecedent sentences
        if chunk_idx > 0:
            antecedent_sentences = self._get_previous_sentences_with_entities(all_sentences, chunk_idx, all_chunks, max_sentences=2)
            if antecedent_sentences:
                # Check if we already have these sentences (avoid duplication)
                existing_sentences = set(chunk_sentences)
                new_sentences = [sent for sent in antecedent_sentences if sent not in existing_sentences]
                
                if new_sentences:
                    new_text = " ".join(new_sentences + chunk_sentences)
                    return self._create_chunk(new_text, chunk.start_idx, chunk.metadata['chunk_id'])
        
        return chunk
    
    def _apply_sentence_start_rule(self, chunk: Chunk, all_chunks: List[Chunk], all_sentences: List[str], chunk_idx: int) -> Chunk:
        """Apply sentence start rule: add context for sentences starting with He/She/They/This/That/It."""
        chunk_sentences = self._split_into_sentences(chunk.text)
        if not chunk_sentences:
            return chunk
        
        # Check for sentences starting with specific pronouns
        hazard_pronouns = {'he', 'she', 'they', 'this', 'that', 'it'}
        
        has_pronoun_start = False
        for sentence in chunk_sentences:
            first_word = sentence.strip().split()[0].lower() if sentence.strip() else ""
            if first_word in hazard_pronouns:
                has_pronoun_start = True
                break
        
        if not has_pronoun_start:
            return chunk
        
        # Include one extra antecedent sentence
        if chunk_idx > 0:
            antecedent_sentences = self._get_previous_sentences_with_entities(all_sentences, chunk_idx, all_chunks, max_sentences=1)
            if antecedent_sentences:
                # Avoid duplication
                unique_antecedents = [sent for sent in antecedent_sentences if sent not in chunk_sentences]
                if unique_antecedents:
                    new_text = " ".join(unique_antecedents + chunk_sentences)
                    return self._create_chunk(new_text, chunk.start_idx, chunk.metadata['chunk_id'])
        
        return chunk
    
    def _contains_pronouns(self, text: str) -> bool:
        """Check if text contains pronouns."""
        doc = self.nlp(text)
        all_pronouns = set()
        for pronoun_type in self.PRONOUNS.values():
            all_pronouns.update(pronoun_type)
        
        for token in doc:
            if token.lemma_.lower() in all_pronouns or token.pos_ == "PRON":
                return True
        return False
    
    def _get_previous_sentences_with_entities(self, all_sentences: List[str], chunk_idx: int, all_chunks: List[Chunk], max_sentences: int = 1) -> List[str]:
        """Get previous sentences that contain named entities by looking back through the original sentence sequence."""
        if chunk_idx == 0:
            return []
        
        # Find the starting sentence index for the current chunk
        current_chunk = all_chunks[chunk_idx] if chunk_idx < len(all_chunks) else None
        current_chunk_sentences = self._split_into_sentences(current_chunk.text) if current_chunk else []
        
        # Find where current chunk starts in the original sentence sequence
        current_start_idx = None
        if current_chunk_sentences:
            first_sentence = current_chunk_sentences[0]
            for i, sent in enumerate(all_sentences):
                if sent.strip() == first_sentence.strip():
                    current_start_idx = i
                    break
        
        if current_start_idx is None:
            # Fallback to chunk-based approach
            current_start_idx = chunk_idx * 2  # Rough estimate
        
        # Look backwards through original sentences to find entities
        result = []
        for i in range(current_start_idx - 1, -1, -1):
            if i < len(all_sentences):
                sentence = all_sentences[i]
                if self._has_named_entities(sentence):
                    result.insert(0, sentence)  # Insert at beginning to maintain order
                    if len(result) >= max_sentences:
                        return result
        
        # Fallback: if no entities found, get recent sentences with concrete nouns
        if not result:
            for i in range(current_start_idx - 1, max(0, current_start_idx - 6), -1):
                if i < len(all_sentences):
                    sentence = all_sentences[i]
                    if self._has_concrete_nouns(sentence):
                        result.insert(0, sentence)
                        if len(result) >= max_sentences:
                            return result
        
        return result
    
    def _has_concrete_nouns(self, text: str) -> bool:
        """Check if text contains concrete nouns (fallback when no named entities)."""
        doc = self.nlp(text)
        for token in doc:
            if token.pos_ == "NOUN" and not token.lemma_.lower() in {'thing', 'person', 'place', 'way', 'time'}:
                return True
        return False