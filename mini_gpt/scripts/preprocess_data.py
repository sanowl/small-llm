# scripts/preprocess_data.py

import os
from utils.data_utils import load_scientific_papers_dataset, preprocess_dataset, save_processed_data

def main():
    # Load raw dataset
    dataset = load_scientific_papers_dataset()
    
    # Preprocess dataset (e.g., sample 50,000 for training)
    processed_dataset = preprocess_dataset(dataset, sample_size=50000)
    
    # Ensure output directory exists
    os.makedirs('data/processed/', exist_ok=True)
    
    # Save processed dataset
    save_processed_data(processed_dataset, output_dir='data/processed/')
    print("Data preprocessing completed and saved to data/processed/")

if __name__ == "__main__":
    main()
