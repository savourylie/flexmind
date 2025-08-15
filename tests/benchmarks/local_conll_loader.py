"""
Local CoNLL-2003 dataset loader for rigorous NER benchmarking.

Loads the full CoNLL-2003 dataset from local Parquet files for comprehensive 
evaluation against established benchmarks.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from functools import lru_cache

from .benchmark_entity_extractor import AnnotatedExample

logger = logging.getLogger(__name__)


@lru_cache(maxsize=3)
def load_conll2003_local(split: str = "test", max_examples: Optional[int] = None) -> List[AnnotatedExample]:
    """
    Load CoNLL-2003 dataset from local Parquet files.
    
    Args:
        split: Dataset split ("train", "dev", "test")
        max_examples: Limit number of examples (None for all)
        
    Returns:
        List of AnnotatedExample objects
        
    Dataset sizes:
        - Train: ~14,000 sentences
        - Dev: ~3,200 sentences  
        - Test: ~3,400 sentences
        - Entity types: PER, LOC, ORG, MISC
    """
    data_dir = Path(__file__).parent.parent.parent / "data" / "colnn2003"
    
    # Map split names to file names
    split_files = {
        "train": "train.parquet",
        "dev": "dev.parquet", 
        "validation": "dev.parquet",  # Alternative name
        "test": "test.parquet"
    }
    
    if split not in split_files:
        raise ValueError(f"Unknown split '{split}'. Available: {list(split_files.keys())}")
    
    parquet_file = data_dir / split_files[split]
    
    if not parquet_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {parquet_file}")
    
    print(f"Loading CoNLL-2003 {split} dataset from {parquet_file}")
    
    try:
        # Load the parquet file
        df = pd.read_parquet(parquet_file)
        print(f"Loaded {len(df)} rows from parquet file")
        
        if max_examples:
            df = df.head(max_examples)
            print(f"Limited to {len(df)} examples")
        
        # Convert to AnnotatedExample objects
        examples = []
        
        for i, row in df.iterrows():
            if i % 1000 == 0:
                print(f"  Processing row {i+1}/{len(df)}")
            
            example = _convert_parquet_row_to_annotated(row, i)
            if example:
                examples.append(example)
        
        print(f"Converted {len(examples)} examples from {split} split")
        return examples
        
    except Exception as e:
        logger.error(f"Failed to load CoNLL-2003 dataset from {parquet_file}: {e}")
        raise


def _convert_parquet_row_to_annotated(row: pd.Series, row_id: int) -> Optional[AnnotatedExample]:
    """Convert a parquet row to AnnotatedExample format."""
    
    # The exact column names depend on how the parquet was created
    # Let's inspect the row to understand the structure
    possible_token_cols = ['tokens', 'words', 'token', 'word']
    possible_tag_cols = ['ner_tags', 'labels', 'tags', 'ner', 'label']
    
    tokens = None
    ner_tags = None
    
    # Find token column
    for col in possible_token_cols:
        if col in row.index and row[col] is not None:
            tokens = row[col]
            break
    
    # Find NER tags column  
    for col in possible_tag_cols:
        if col in row.index and row[col] is not None:
            ner_tags = row[col]
            break
    
    # Check if we found the required columns
    if tokens is None or ner_tags is None:
        logger.warning(f"Could not find tokens/tags in row {row_id}. Columns: {list(row.index)}")
        return None
    
    # Handle different data formats
    if isinstance(tokens, str):
        # If tokens are a string, try to parse them
        try:
            tokens = eval(tokens) if tokens.startswith('[') else tokens.split()
        except:
            tokens = tokens.split()
    
    if isinstance(ner_tags, str):
        # If tags are a string, try to parse them
        try:
            ner_tags = eval(ner_tags) if ner_tags.startswith('[') else ner_tags.split()
        except:
            ner_tags = ner_tags.split()
    
    # Convert numpy arrays to lists if needed
    if hasattr(tokens, 'tolist'):
        tokens = tokens.tolist()
    if hasattr(ner_tags, 'tolist'):
        ner_tags = ner_tags.tolist()
    
    # Check if we have valid data
    if len(tokens) == 0 or len(ner_tags) == 0:
        return None
    
    if len(tokens) != len(ner_tags):
        logger.warning(f"Token/tag length mismatch in row {row_id}: {len(tokens)} tokens, {len(ner_tags)} tags")
        return None
    
    # Reconstruct text from tokens
    text = ' '.join(str(token) for token in tokens)
    
    # Convert numeric tags to labels if needed
    if all(isinstance(tag, (int, float)) for tag in ner_tags):
        # Numeric tags - convert to BIO labels
        label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        ner_tags = [label_names[int(tag)] if int(tag) < len(label_names) else 'O' for tag in ner_tags]
    
    # Extract entities from BIO tags
    entities = []
    current_entity = None
    char_pos = 0
    
    for token, tag in zip(tokens, ner_tags):
        token = str(token)
        token_start = char_pos
        token_end = char_pos + len(token)
        
        if isinstance(tag, str) and tag.startswith('B-'):
            # Finish previous entity
            if current_entity:
                entities.append((
                    current_entity['text'],
                    current_entity['type'],
                    current_entity['start'],
                    current_entity['end']
                ))
            
            # Start new entity
            entity_type = tag[2:]
            current_entity = {
                'text': token,
                'type': entity_type,
                'start': token_start,
                'end': token_end
            }
        elif isinstance(tag, str) and tag.startswith('I-') and current_entity and tag[2:] == current_entity['type']:
            # Continue current entity
            current_entity['text'] += ' ' + token
            current_entity['end'] = token_end
        else:
            # O tag or different entity - finish current entity
            if current_entity:
                entities.append((
                    current_entity['text'],
                    current_entity['type'],
                    current_entity['start'],
                    current_entity['end']
                ))
                current_entity = None
        
        char_pos = token_end + 1  # +1 for space
    
    # Don't forget final entity
    if current_entity:
        entities.append((
            current_entity['text'],
            current_entity['type'],
            current_entity['start'],
            current_entity['end']
        ))
    
    return AnnotatedExample(
        text=text,
        entities=entities,
        metadata={'source': 'conll2003_local', 'row_id': row_id}
    )


def inspect_parquet_structure():
    """Inspect the structure of the CoNLL-2003 parquet files."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "colnn2003"
    
    print("CoNLL-2003 Parquet File Structure:")
    print("=" * 50)
    
    for split_file in ["train.parquet", "dev.parquet", "test.parquet"]:
        file_path = data_dir / split_file
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                print(f"\n{split_file}:")
                print(f"  Rows: {len(df)}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Sample row (first few columns):")
                
                # Show first row sample
                for col in df.columns[:5]:  # Show first 5 columns
                    sample_val = df[col].iloc[0]
                    if isinstance(sample_val, (list, tuple)):
                        print(f"    {col}: {sample_val[:3]}... (length: {len(sample_val)})")
                    else:
                        print(f"    {col}: {str(sample_val)[:50]}...")
                        
            except Exception as e:
                print(f"  Error reading {split_file}: {e}")
        else:
            print(f"  {split_file}: NOT FOUND")


def get_dataset_stats():
    """Get statistics about the local CoNLL-2003 dataset."""
    stats = {}
    
    for split in ["train", "dev", "test"]:
        try:
            examples = load_conll2003_local(split)
            
            entity_counts = {}
            total_entities = 0
            total_tokens = 0
            
            for example in examples:
                total_tokens += len(example.text.split())
                total_entities += len(example.entities)
                
                for _, entity_type, _, _ in example.entities:
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            stats[split] = {
                'sentences': len(examples),
                'total_entities': total_entities,
                'total_tokens': total_tokens,
                'entities_per_sentence': total_entities / len(examples) if examples else 0,
                'tokens_per_sentence': total_tokens / len(examples) if examples else 0,
                'entity_type_counts': entity_counts
            }
            
        except Exception as e:
            stats[split] = {'error': str(e)}
    
    return stats


if __name__ == "__main__":
    # Inspect the parquet structure first
    print("Inspecting CoNLL-2003 parquet files...")
    inspect_parquet_structure()
    
    print("\n" + "=" * 50)
    print("Testing dataset loading...")
    
    # Try to load a small sample
    try:
        sample_data = load_conll2003_local("test", max_examples=5)
        print(f"\nSuccessfully loaded {len(sample_data)} test examples:")
        
        for i, example in enumerate(sample_data):
            print(f"\nExample {i+1}:")
            print(f"  Text: {example.text[:100]}...")
            print(f"  Entities: {[(e[0], e[1]) for e in example.entities]}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
    
    print("\n" + "=" * 50)
    print("Dataset statistics:")
    
    stats = get_dataset_stats()
    for split, split_stats in stats.items():
        print(f"\n{split.upper()}:")
        if 'error' in split_stats:
            print(f"  Error: {split_stats['error']}")
        else:
            print(f"  Sentences: {split_stats['sentences']:,}")
            print(f"  Total entities: {split_stats['total_entities']:,}")
            print(f"  Entities per sentence: {split_stats['entities_per_sentence']:.2f}")
            print(f"  Entity types: {split_stats['entity_type_counts']}")