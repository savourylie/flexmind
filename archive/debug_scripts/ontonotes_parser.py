#!/usr/bin/env python3
"""Parse OntoNotes CoNLL skeleton files and extract text and coreference information."""

import os
import glob
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

class OntoNotesDocument:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.sentences = []
        self.clusters = defaultdict(list)  # cluster_id -> list of (sent_idx, start, end)
        
    def add_sentence(self, tokens: List[str], coref_spans: List[Tuple[int, int, int]]):
        """Add a sentence with tokens and coreference spans."""
        sent_idx = len(self.sentences)
        self.sentences.append(tokens)
        
        # Add coreference information
        for cluster_id, start, end in coref_spans:
            self.clusters[cluster_id].append((sent_idx, start, end))
    
    def get_text(self) -> str:
        """Get the full document text."""
        return " ".join([" ".join(tokens) for tokens in self.sentences])
    
    def get_coref_clusters(self) -> List[List[str]]:
        """Get coreference clusters as lists of text spans."""
        clusters = []
        for cluster_id, spans in self.clusters.items():
            cluster_texts = []
            for sent_idx, start, end in spans:
                if sent_idx < len(self.sentences):
                    tokens = self.sentences[sent_idx]
                    if start < len(tokens) and end < len(tokens):
                        span_text = " ".join(tokens[start:end+1])
                        cluster_texts.append(span_text)
            if cluster_texts:
                clusters.append(cluster_texts)
        return clusters

def parse_skeleton_file(filepath: str) -> Optional[OntoNotesDocument]:
    """Parse a single OntoNotes skeleton file."""
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Extract document ID from first line
        doc_id = None
        for line in lines:
            if line.startswith('#begin document'):
                # Extract document ID from "#begin document (bc/cctv/00/cctv_0005)"
                match = re.search(r'\((.*?)\)', line)
                if match:
                    doc_id = match.group(1)
                break
        
        if not doc_id:
            return None
            
        document = OntoNotesDocument(doc_id)
        
        # Group lines by sentence
        current_sentence = []
        sentence_coref = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split('\t')
            if len(parts) < 12:  # CoNLL format should have at least 12 columns
                continue
                
            # Extract token and coref information
            # Format: doc_id sent_id token_id [WORD] pos parse ... ... ... speaker ... ... coref_column
            if parts[3] == '[WORD]':  # This is a placeholder - we need real words
                continue
                
            token = parts[3]
            coref_column = parts[-1]  # Last column contains coreference info
            
            current_sentence.append(token)
            
            # Parse coreference information 
            coref_spans = parse_coref_column(coref_column, len(current_sentence) - 1)
            sentence_coref.extend(coref_spans)
            
            # Check if sentence ends (you'd need to implement sentence boundary detection)
            # For now, let's assume each line is a token and we'll group them heuristically
            
        # Add the last sentence if any
        if current_sentence:
            document.add_sentence(current_sentence, sentence_coref)
            
        return document
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def parse_coref_column(coref_col: str, token_idx: int) -> List[Tuple[int, int, int]]:
    """Parse the coreference column and extract cluster information."""
    spans = []
    
    if coref_col == '-':
        return spans
    
    # Parse patterns like "(42)", "42)", "(32", etc.
    # (42) means start and end of entity 42 at this token
    # (42 means start of entity 42
    # 42) means end of entity 42
    
    # Find all cluster mentions
    patterns = re.findall(r'\((\d+)\)|(\d+)\)|\((\d+)', coref_col)
    
    for match in patterns:
        if match[0]:  # (42) - single token entity
            cluster_id = int(match[0])
            spans.append((cluster_id, token_idx, token_idx))
        elif match[1]:  # 42) - end of entity
            cluster_id = int(match[1]) 
            # We'd need to track starts to properly handle this
            # For now, just note it
            pass
        elif match[2]:  # (42 - start of entity
            cluster_id = int(match[2])
            # We'd need to track this until we see the end
            pass
    
    return spans

def load_ontonotes_sample(data_dir: str, split: str = "development", max_files: int = 5) -> List[OntoNotesDocument]:
    """Load a sample of OntoNotes documents for testing."""
    
    split_dir = os.path.join(data_dir, f"conll-formatted-ontonotes-5.0", "data", split, "data", "english", "annotations")
    
    if not os.path.exists(split_dir):
        print(f"Directory not found: {split_dir}")
        return []
    
    # Find skeleton files
    pattern = os.path.join(split_dir, "**", "*.gold_skel")
    skel_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(skel_files)} skeleton files in {split}")
    
    documents = []
    for skel_file in skel_files[:max_files]:
        print(f"Processing: {skel_file}")
        doc = parse_skeleton_file(skel_file)
        if doc:
            documents.append(doc)
        
    return documents

def test_ontonotes_parsing():
    """Test the OntoNotes parsing functionality."""
    
    print("=== Testing OntoNotes Parsing ===")
    
    data_dir = "/Users/calvinku/FunProjects/flexmind/data"
    
    # Load a small sample
    documents = load_ontonotes_sample(data_dir, split="development", max_files=2)
    
    if not documents:
        print("No documents loaded - check data path and file format")
        return
    
    print(f"Loaded {len(documents)} documents")
    
    for i, doc in enumerate(documents):
        print(f"\n--- Document {i+1}: {doc.doc_id} ---")
        text = doc.get_text()
        print(f"Text length: {len(text)} characters")
        print(f"Text preview: {text[:200]}...")
        
        clusters = doc.get_coref_clusters()
        print(f"Found {len(clusters)} coreference clusters")
        
        for j, cluster in enumerate(clusters[:3]):  # Show first 3 clusters
            print(f"  Cluster {j+1}: {cluster}")

if __name__ == "__main__":
    test_ontonotes_parsing()