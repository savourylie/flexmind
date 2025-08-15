"""
CoNLL-2003 format data loader for standard NER benchmarking.

Supports loading datasets in CoNLL-2003 format for rigorous evaluation
against established benchmarks.
"""

from pathlib import Path
from typing import List, Iterator, Tuple
from .benchmark_entity_extractor import AnnotatedExample


def load_conll_file(filepath: Path) -> List[AnnotatedExample]:
    """
    Load a CoNLL-2003 format file into AnnotatedExample objects.
    
    CoNLL-2003 format:
    word1 POS1 chunk1 NER1
    word2 POS2 chunk2 NER2
    ...
    (blank line separates sentences)
    
    Args:
        filepath: Path to CoNLL format file
        
    Returns:
        List of AnnotatedExample objects
    """
    if not filepath.exists():
        return []
    
    examples = []
    current_tokens = []
    current_labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line == '' or line.startswith('-DOCSTART-'):
                # End of sentence - process current tokens
                if current_tokens:
                    example = _tokens_to_example(current_tokens, current_labels)
                    if example:
                        examples.append(example)
                    current_tokens = []
                    current_labels = []
            else:
                # Parse token line
                parts = line.split()
                if len(parts) >= 4:
                    token = parts[0]
                    ner_label = parts[3]  # NER label is 4th column
                    
                    current_tokens.append(token)
                    current_labels.append(ner_label)
    
    # Process final sentence if file doesn't end with blank line
    if current_tokens:
        example = _tokens_to_example(current_tokens, current_labels)
        if example:
            examples.append(example)
    
    return examples


def _tokens_to_example(tokens: List[str], labels: List[str]) -> AnnotatedExample:
    """Convert tokenized sentence to AnnotatedExample."""
    if not tokens:
        return None
    
    # Reconstruct text (simple space joining)
    text = ' '.join(tokens)
    
    # Extract entities from BIO labels
    entities = []
    current_entity = None
    current_start = 0
    char_pos = 0
    
    for token, label in zip(tokens, labels):
        token_start = char_pos
        token_end = char_pos + len(token)
        
        if label.startswith('B-'):
            # Beginning of new entity
            entity_type = label[2:]  # Remove 'B-' prefix
            current_entity = {
                'text': token,
                'type': entity_type,
                'start': token_start,
                'end': token_end
            }
        elif label.startswith('I-') and current_entity:
            # Inside entity - extend current entity
            current_entity['text'] += ' ' + token
            current_entity['end'] = token_end
        else:
            # O label or start of new entity - finish current entity
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
    
    return AnnotatedExample(text=text, entities=entities)


def create_mini_conll_dataset() -> List[AnnotatedExample]:
    """
    Create a small CoNLL-style dataset for testing.
    
    This simulates what a real CoNLL dataset would look like but with
    examples relevant to our use case.
    """
    return [
        # Sentence 1: Tech company news
        AnnotatedExample(
            text="Apple CEO Tim Cook announced quarterly results in Cupertino yesterday.",
            entities=[
                ("Apple", "ORG", 0, 5),
                ("Tim Cook", "PER", 10, 18), 
                ("Cupertino", "LOC", 55, 64)
            ]
        ),
        
        # Sentence 2: Meeting/business
        AnnotatedExample(
            text="Microsoft President Brad Smith met with European Union officials in Brussels.",
            entities=[
                ("Microsoft", "ORG", 0, 9),
                ("Brad Smith", "PER", 20, 30),
                ("European Union", "ORG", 40, 54),
                ("Brussels", "LOC", 68, 76)
            ]
        ),
        
        # Sentence 3: Investment/money
        AnnotatedExample(
            text="Google invested fifty million dollars in the artificial intelligence startup DeepMind.",
            entities=[
                ("Google", "ORG", 0, 6),
                ("fifty million dollars", "MONEY", 16, 38),
                ("DeepMind", "ORG", 76, 84)
            ]
        ),
        
        # Sentence 4: Research/academic  
        AnnotatedExample(
            text="Dr. Fei-Fei Li from Stanford University published groundbreaking research on computer vision.",
            entities=[
                ("Dr. Fei-Fei Li", "PER", 0, 14),
                ("Stanford University", "ORG", 20, 39)
            ]
        ),
        
        # Sentence 5: Government/policy
        AnnotatedExample(
            text="President Biden signed the AI Safety Act in Washington D.C. on March 15th 2024.", 
            entities=[
                ("Biden", "PER", 10, 15),
                ("AI Safety Act", "MISC", 27, 40),
                ("Washington D.C.", "LOC", 44, 59),
                ("March 15th 2024", "DATE", 63, 78)
            ]
        )
    ]


def save_conll_format(examples: List[AnnotatedExample], filepath: Path):
    """
    Save examples in CoNLL-2003 format for external benchmarking tools.
    
    This is useful for comparing with other NER systems using standard tools.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for example in examples:
            # Convert back to BIO format (simplified)
            tokens = example.text.split()
            labels = ['O'] * len(tokens)
            
            # Mark entities with BIO labels
            for entity_text, entity_type, start, end in example.entities:
                entity_tokens = entity_text.split()
                
                # Find token positions (simplified - assumes space separation)
                for i, token in enumerate(tokens):
                    if token in entity_tokens:
                        if labels[i] == 'O':  # First token of entity
                            labels[i] = f'B-{entity_type}'
                        else:  # Continuing token
                            labels[i] = f'I-{entity_type}'
            
            # Write in CoNLL format
            for token, label in zip(tokens, labels):
                f.write(f"{token} _ _ {label}\n")
            f.write("\n")  # Blank line between sentences


if __name__ == "__main__":
    # Test the CoNLL data loader
    print("Creating mini CoNLL dataset...")
    mini_dataset = create_mini_conll_dataset()
    
    print(f"Created {len(mini_dataset)} examples:")
    for i, example in enumerate(mini_dataset):
        print(f"  {i+1}. {example.text[:60]}...")
        print(f"     Entities: {[f'{text}({label})' for text, label, _, _ in example.entities]}")
    
    # Save in CoNLL format for external tools
    conll_file = Path("tests/benchmarks/data/mini_conll.txt")
    save_conll_format(mini_dataset, conll_file)
    print(f"\nSaved CoNLL format data to: {conll_file}")