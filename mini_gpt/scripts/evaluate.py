# scripts/evaluate.py

import torch
from models.mini_gpt import MiniGPT
from tokenizers.tokenizer import load_trained_tokenizer
from utils.data_utils import load_scientific_papers_dataset
from utils.training_utils import get_dataloader
from utils.evaluation_utils import calculate_perplexity, evaluate_generation
from datasets import load_from_disk

def main():
    # Configuration
    model_path = 'outputs/models/best_mini_gpt.pth'
    tokenizer_path = 'outputs/tokenizer/'
    processed_data_path = 'data/processed/'
    batch_size = 8
    num_samples = 100  # For generation evaluation
    
    # Load tokenizer
    tokenizer = load_trained_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer)
    
    # Load processed dataset
    dataset = load_from_disk(processed_data_path)
    valid_dataset = dataset['validation']
    
    # Tokenize validation dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True
        )
    
    tokenized_valid = valid_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    # Create DataLoader
    valid_dataloader = get_dataloader(tokenized_valid, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_size=512,
        num_heads=8,
        hidden_dim=2048,
        num_layers=12,
        max_seq_length=512,
        dropout=0.1
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Define loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Calculate Perplexity
    perplexity = calculate_perplexity(model, valid_dataloader, criterion, device)
    print(f"Validation Perplexity: {perplexity}")
    
    # Evaluate Generation
    rouge_scores = evaluate_generation(model, tokenizer, valid_dataloader, device, num_samples=num_samples)
    print("ROUGE Scores:", rouge_scores)

if __name__ == "__main__":
    main()
