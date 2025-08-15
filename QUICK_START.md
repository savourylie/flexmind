# FlexMind Quick Start Guide

## Installation & Setup
```bash
# Install dependencies
uv sync

# Test the implementations
uv run pytest tests/unit/ -v
```

## Usage Examples

### 1. TextPreprocessor
```python
from flexmind.core.preprocessing.text import TextPreprocessor

# Initialize
preprocessor = TextPreprocessor(window_size=5, max_tokens=512)

# Dialog processing (sliding window)
dialog = """
Alice: I work at OpenAI!
Bob: That's cool! What do you do?
Alice: I'm a researcher.
"""
chunks = preprocessor.process(dialog, text_type='dialog')
for chunk in chunks:
    print(chunk)  # DialogChunk(turns=0-2, text='Alice: I work at...')

# Document processing (sentence-based)
doc = "AI systems use neural networks. They process language efficiently."
doc_chunks = preprocessor.process(doc, text_type='document')
for chunk in doc_chunks:
    print(chunk)  # DocumentChunk(tokens=10, sentences=2, text='AI systems...')
```

### 2. EntityExtractor
```python
from flexmind.core.extractors.entities import EntityExtractor

# Initialize
extractor = EntityExtractor(confidence_threshold=0.75, use_fallback=True)

# Extract entities
text = "Alice Johnson works at Microsoft in San Francisco since 2024."
entities = extractor.extract(text)

for entity in entities:
    print(entity)  # Entity('Alice Johnson' → PERSON)
    print(f"  Confidence: {entity.confidence}")
    print(f"  Source: {entity.source}")  # 'spacy' or 'hybrid'
```

## Interactive Demos

### Try TextPreprocessor
```bash
uv run python demo_text_preprocessor.py
```
**Features:**
- Dialog chunking with sliding windows
- Document sentence chunking  
- Interactive mode to test your own text
- Configuration examples

### Try EntityExtractor
```bash
uv run python demo_entity_extractor.py
```
**Features:**
- Basic entity extraction
- spaCy vs DistilBERT fallback comparison
- Confidence filtering examples
- Interactive testing mode

## Clear Output Format

### TextChunk Objects
```python
DialogChunk(turns=0-2, text='Alice: I work at OpenAI...')
DocumentChunk(tokens=15, sentences=2, text='The system processes...')
```

### Entity Objects
```python
Entity('Alice Johnson' → PERSON)
Entity('OpenAI' → ORG [hybrid])  # Shows fallback source
Entity('San Francisco' → GPE (conf: 0.82))  # Shows confidence when <1.0
```

## What to Test

1. **Dialog Examples:**
   ```
   Alice: I just started at OpenAI!
   Bob: When did you move to San Francisco?  
   Alice: Last month. Working on GPT-5 now.
   ```

2. **Complex Entities:**
   ```
   Dr. Yann LeCun leads FAIR at Meta in Menlo Park.
   President Biden met with Tesla CEO Elon Musk.
   ```

3. **Performance:**
   - First run: ~7-10s (model loading)
   - Subsequent runs: <100ms
   - Processing: 50k-100k tokens/sec

## Next Steps
Ready to continue with **RelationExtractor** to complete Day 1 of the TDD implementation!