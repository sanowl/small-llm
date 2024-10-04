# scripts/train.py

import os
import torch
from models.mini_gpt import MiniGPT
from tokenizers.tokenizer import load_trained_tokenizer
from utils.data_utils import load_scientific_papers_dataset
from utils.training_utils import (
    get_dataloader,
    train_epoch,
    evaluate,
    setup_optimizer_scheduler,
    log_metrics
)
from utils.evaluation_utils import calculate_perplexity
from datasets import load_from_disk
import mlflow

def main():
    # Configuration
    processed_data_path = 'data/processed/'
    tokenizer_path = 'outputs/tokenizer/'
    model_save_path = 'outputs/models/best_mini_gpt.pth'
    batch_size = 8
    epochs = 10
    learning_rate = 5e-4
    weight_decay = 1e-2
    gradient_accumulation_steps = 4
    patience = 3
    
    # Initialize MLflow
    mlflow.start_run()
    
    # Load tokenizer
    tokenizer = load_trained_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer)
    
    # Load processed dataset
    dataset = load_from_disk(processed_data_path)
    train_dataset = dataset['train']
    valid_dataset = dataset['validation']
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True
        )
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized_valid = valid_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    # Create DataLoader objects
    train_dataloader = get_dataloader(tokenized_train, batch_size=batch_size, shuffle=True, num_workers=4)
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
    
    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer, scheduler = setup_optimizer_scheduler(model, learning_rate, weight_decay, epochs)
    
    # Initialize GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else torch.cuda.amp.GradScaler(enabled=False)
    
    # Log hyperparameters
    mlflow.log_param("embed_size", 512)
    mlflow.log_param("num_heads", 8)
    mlflow.log_param("hidden_dim", 2048)
    mlflow.log_param("num_layers", 12)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("weight_decay", weight_decay)
    mlflow.log_param("epochs", epochs)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    trigger_times = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        train_loss = train_epoch(
            model, 
            train_dataloader, 
            criterion, 
            optimizer, 
            scaler, 
            device, 
            gradient_accumulation_steps
        )
        print(f"Training Loss: {train_loss}")
        
        # Validate
        val_loss, perplexity = evaluate(model, valid_dataloader, criterion, device)
        print(f"Validation Loss: {val_loss} | Perplexity: {perplexity}")
        
        # Log metrics
        log_metrics(epoch, train_loss, val_loss, perplexity)
        
        # Scheduler step
        scheduler.step()
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
            mlflow.pytorch.log_model(model, "best_model")
            print("Best model updated and saved.")
        else:
            trigger_times += 1
            print(f"No improvement in validation loss for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break
    
    mlflow.end_run()
    print("Training completed.")

if __name__ == "__main__":
    main()
