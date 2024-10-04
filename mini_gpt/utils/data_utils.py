# utils/data_utils.py

import re
from datasets import load_dataset, Dataset
from tokenizers import ByteLevelBPETokenizer

def load_scientific_papers_dataset():
    dataset = load_dataset('scientific_papers')
    return dataset

def combine_fields(example):
    return {'text': example['abstract'] + ' ' + ' '.join(example['body_text'])}

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters except standard punctuation
    text = re.sub(r'[^A-Za-z0-9\s\.,;:\'\"\!\?()-]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataset(dataset, sample_size=None):
    # Combine fields
    dataset = dataset.map(combine_fields, remove_columns=['abstract', 'body_text', 'title', 'url'])
    # Clean text
    dataset = dataset.map(lambda x: {'text': clean_text(x['text'])}, batched=True)
    # Sample if needed
    if sample_size:
        dataset['train'] = dataset['train'].shuffle(seed=42).select(range(sample_size))
    return dataset

def save_processed_data(dataset, output_dir='data/processed/'):
    dataset.save_to_disk(output_dir)
