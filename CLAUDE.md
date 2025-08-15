# FlexMind - Lightweight Knowledge Graph Memory System

## Project Overview
A human-like memory system for LLMs using hybrid architecture: knowledge graph for dynamic structured facts + vector store for static unstructured knowledge.

**Core Principles:**
- Speed First: CPU-optimized processing under 60 seconds
- Incremental Updates: No full graph rebuilds
- Cost Efficient: Minimal LLM usage, dependency-based extraction
- Temporal Awareness: Track when facts change over time

## 4-Day TDD Implementation Plan

### Day 1: Core Pipeline + NER/Relation Extraction
**TDD Focus:** Entity and relation extraction with comprehensive test coverage

**Implementation:**
- Text preprocessing with sliding window for dialogs
- Named Entity Recognition using spaCy (primary) + DistilBERT (fallback)
- Relation extraction via dependency parsing
- Basic entity resolution with embedding similarity

**Tests Required:**
- Unit tests for text preprocessing (dialog vs document chunking)
- NER accuracy tests with known entities
- Relation extraction tests with dependency patterns
- Entity resolution similarity matching tests

**Performance Target:** 100 sentences/sec NER, 50 sentences/sec relations

### Day 2: Neo4j Integration + Temporal Updates
**TDD Focus:** Knowledge graph storage with temporal conflict resolution

**Implementation:**
- Neo4j graph database integration
- Temporal entity resolver with conflict detection
- Entity/relation update mechanisms with timestamps
- Basic query interface for current facts vs history

**Tests Required:**
- Neo4j connection and CRUD operation tests
- Temporal conflict resolution test scenarios
- Entity update and versioning tests
- Query interface tests for time-based retrieval

**Performance Target:** 66 updates/sec to graph

### Day 3: Vector Store + Hybrid Retrieval
**TDD Focus:** Unstructured content storage and hybrid memory retrieval

**Implementation:**
- ChromaDB vector store integration
- Embedding generation and caching
- Hybrid retrieval combining graph + vector results
- Context merging and ranking algorithms

**Tests Required:**
- Vector store embedding and retrieval tests
- Hybrid query result combination tests
- Semantic search accuracy tests
- Context ranking and relevance tests

**Performance Target:** Sub-second hybrid query responses

### Day 4: Performance Optimization + API
**TDD Focus:** Production-ready API with performance benchmarks

**Implementation:**
- FastAPI REST interface
- Caching strategy (Redis integration)
- Batch processing optimizations
- Monitoring and performance metrics

**Tests Required:**
- API endpoint integration tests
- Performance benchmark tests
- Cache hit/miss ratio tests
- End-to-end system load tests

**Performance Target:** <60s total pipeline processing

## Technical Stack

### Core Dependencies
```bash
# Install with uv
uv add spacy transformers neo4j chromadb fastapi redis
uv add pytest pytest-benchmark pytest-asyncio  # Testing
uv add numpy scikit-learn sentence-transformers  # ML utilities

# Download spaCy model
uv run python -m spacy download en_core_web_lg
```

### Database Setup
```bash
# Neo4j (local development)
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

# Redis (caching)
docker run -d --name redis -p 6379:6379 redis:latest
```

## Key Architecture Components

### 1. Entity Extraction Pipeline
- **Primary:** spaCy (100k tokens/sec)
- **Fallback:** DistilBERT (60% faster than BERT, 97% accuracy)
- **Confidence threshold:** 0.85 for entity resolution

### 2. Relation Extraction
- Dependency parsing with pattern matching
- Rule-based extraction from dependency trees
- 94% quality of LLM-based extraction at 100 sentences/sec

### 3. Knowledge Graph Schema
```cypher
# Entity types: Person, Organization, Location, Event, Concept
# Relationships: works_for, located_in, participates_in, related_to
# Temporal properties: valid_from, valid_until, confidence_score, source
```

### 4. Temporal Conflict Resolution
- Track fact changes over time
- Expire contradictory facts with timestamps
- Maintain entity version history

## Testing Strategy

### Unit Tests
```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Test specific components
uv run pytest tests/unit/test_entity_extractor.py -v
uv run pytest tests/unit/test_relation_extractor.py -v
uv run pytest tests/unit/test_temporal_resolver.py -v
```

### Integration Tests
```bash
# Database integration
uv run pytest tests/integration/test_neo4j_integration.py -v
uv run pytest tests/integration/test_vector_store.py -v

# End-to-end pipeline
uv run pytest tests/integration/test_memory_pipeline.py -v
```

### Performance Benchmarks
```bash
# Performance tests with benchmarks
uv run pytest tests/performance/ --benchmark-only

# Specific performance targets
uv run pytest tests/performance/test_processing_speed.py --benchmark-min-rounds=10
```

## Development Commands

### Running the System
```bash
# Start the memory system API
uv run python -m flexmind.api.main

# Process text through pipeline
uv run python -m flexmind.scripts.process_text --input "sample.txt"

# Interactive memory queries
uv run python -m flexmind.scripts.interactive_query
```

### Testing Commands
```bash
# Run all tests with coverage
uv run pytest --cov=flexmind --cov-report=html

# TDD cycle - watch for changes and re-run tests
uv run pytest-watch tests/unit/

# Performance regression testing
uv run pytest tests/performance/ --benchmark-compare=baseline.json
```

## Performance Targets

| Operation | Time | Throughput |
|-----------|------|------------|
| NER (spaCy) | 10ms | 100 sentences/sec |
| Relation Extraction | 20ms | 50 sentences/sec |
| Entity Resolution | 5ms | 200 entities/sec |
| Graph Update | 15ms | 66 updates/sec |
| **Total Pipeline** | <60s | 1000 sentences/min |

## Example Usage

```python
# Initialize the memory system
from flexmind.core.memory import KnowledgeGraphMemory

memory = KnowledgeGraphMemory()

# Process dialog
dialog = """
Alice: I just started working at OpenAI.
Bob: That's great! When did you move to San Francisco?
Alice: Last month. I'm working on GPT-5 now.
"""

memory.process_dialog(dialog)

# Query facts
facts = memory.query("What does Alice work on?")
# Returns: Alice -> works_on -> GPT-5 (confidence: 0.9, timestamp: now)

# Get entity history
history = memory.get_entity_history("Alice")
# Returns timeline of facts about Alice

# Semantic search
context = memory.semantic_search("Tell me about Alice's career")
# Returns structured facts + relevant document chunks
```

## Project Structure
```
flexmind/
├── flexmind/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── memory.py           # Main memory system
│   │   ├── extractors/
│   │   │   ├── entities.py     # NER with spaCy/DistilBERT
│   │   │   ├── relations.py    # Dependency parsing extraction
│   │   │   └── resolver.py     # Temporal entity resolution
│   │   ├── storage/
│   │   │   ├── graph.py        # Neo4j integration
│   │   │   ├── vector.py       # ChromaDB integration
│   │   │   └── cache.py        # Redis caching
│   │   └── preprocessing/
│   │       ├── text.py         # Text chunking & normalization
│   │       └── dialog.py       # Dialog-specific processing
│   ├── api/
│   │   ├── main.py            # FastAPI application
│   │   ├── routes/
│   │   │   ├── memory.py      # Memory endpoints
│   │   │   └── query.py       # Query endpoints
│   │   └── models/
│   │       └── schemas.py     # Pydantic models
│   └── scripts/
│       ├── process_text.py    # Batch processing
│       └── interactive_query.py # Interactive queries
├── tests/
│   ├── unit/                  # Unit tests for each component
│   ├── integration/           # Database & API integration tests
│   ├── performance/           # Benchmark tests
│   └── fixtures/              # Test data
├── docs/
│   └── SYSTEM_DESIGN.md      # Detailed architecture
├── pyproject.toml            # UV project configuration
└── CLAUDE.md                 # This file
```

## TDD Best Practices
1. **Red-Green-Refactor:** Write failing test → Make it pass → Refactor
2. **Test First:** Every feature starts with a test
3. **Fast Feedback:** Use `pytest-watch` for continuous testing
4. **Performance Tests:** Benchmark critical paths with `pytest-benchmark`
5. **Integration Coverage:** Test database connections and API endpoints
6. **Mocking:** Mock external services (Neo4j, Redis) for unit tests

## Monitoring & Debugging
```bash
# View system metrics
uv run python -m flexmind.scripts.system_metrics

# Debug extraction pipeline
uv run python -m flexmind.scripts.debug_extraction --text "sample text"

# Analyze graph structure
uv run python -m flexmind.scripts.analyze_graph --entity "Alice"
```