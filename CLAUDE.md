# FlexMind - Human-like Memory Mechanism

## Project Vision

This project prototypes a memory mechanism inspired by human memory characteristics from "Why We Remember". The key insight is that human memory prioritizes **flexibility and contextual adaptation** over static, photographic accuracy.

## Current Approach vs. Human-like Memory

### Traditional RAG Limitations
- **Static chunking**: Messages saved in fixed chunks
- **Inflexible retrieval**: Cannot adapt or modify stored information
- **Photographic accuracy focus**: Prioritizes exact recall over contextual relevance

### Human-like Memory Goals
- **Adaptive and flexible**: Memory should evolve with context
- **Entity-relationship focus**: Use knowledge graphs for dynamic connections
- **Context-sensitive**: Prioritize relevance over perfect accuracy
- **Efficient updates**: Quick adaptation without expensive reprocessing

## Technical Challenges

### Cost Considerations
- **Entity relation extraction**: LLM-based extraction is expensive
- **Frequent updates**: Continuous LLM calls for modifications are costly
- **Balance needed**: Human-like flexibility vs. computational efficiency

## Implementation Strategy

Focus on creating a knowledge graph-based memory system that can:
1. Extract and maintain entity relationships efficiently
2. Adapt quickly to new context without full reprocessing  
3. Prioritize contextual relevance over perfect recall
4. Minimize LLM calls while maintaining flexibility

## Guidelines for Development

- Prototype human-like memory characteristics
- Use knowledge graphs for flexible entity-relationship storage
- Optimize for quick adaptation over perfect accuracy
- Find cost-effective alternatives to expensive LLM operations
- Maintain focus on contextual relevance and flexibility
- Always develop in TDD manner.
- Always run scripts  with `uv run`

## Research Ideas

### Late Chunking for Pronoun Resolution (Research Paper Potential)

**Core Idea**: Use contextual token embeddings from late chunking to resolve complex pronoun coreference situations that traditional coref systems struggle with.

**Problem**: Traditional coref fails on cases like "When Alice met Beth at the café, she told her that she had won the lottery" - multiple pronouns referring to different entities in close proximity.

**Proposed Solution**: 
1. Apply late chunking with long-context embedding models (e.g., jina-embeddings-v2-base-en, 8K tokens)
2. Extract contextual token embeddings for pronouns and potential antecedents
3. Use cosine similarity between pronoun embeddings and entity embeddings for resolution
4. Each pronoun embedding is context-aware and should be more similar to its correct antecedent

**Why Novel**: 
- Late chunking (2023) not yet applied to coreference resolution
- Direct token-level similarity approach vs traditional span-based neural coref
- Addresses known hard cases in multi-entity scenarios

**Next Steps When Ready**:
- Design experiments comparing against state-of-the-art coref systems
- Test on OntoNotes + custom challenging multi-entity datasets  
- Analyze performance patterns and failure modes
- Consider hybrid approaches combining with traditional coref
- Target venues: ACL, EMNLP, NAACL

**Date Added**: August 2025