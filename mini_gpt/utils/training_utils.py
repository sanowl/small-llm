# utils/training_utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow

def get_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, gradient_accumulation_steps=4):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        
        with autocast(enabled=True):
            outputs = model(input_ids)
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = criterion(shift_logits.view(-1, model.output_layer.out_features), shift_labels.view(-1))
            loss = loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix({'Loss': loss.item() * gradient_accumulation_steps})
    
    avg_loss = epoch_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            
            with autocast(enabled=True):
                outputs = model(input_ids)
                shift_logits = outputs[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = criterion(shift_logits.view(-1, model.output_layer.out_features), shift_labels.view(-1))
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()

def setup_optimizer_scheduler(model, learning_rate=5e-4, weight_decay=1e-2, epochs=10):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler

def log_metrics(epoch, train_loss, val_loss, perplexity):
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("perplexity", perplexity, step=epoch)
