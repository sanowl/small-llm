# utils/evaluation_utils.py

import math
from tqdm.auto import tqdm
from datasets import load_metric
import torch 

def calculate_perplexity(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Perplexity"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(input_ids)
                shift_logits = outputs[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = criterion(shift_logits.view(-1, model.output_layer.out_features), shift_labels.view(-1))
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return perplexity

def evaluate_generation(model, tokenizer, dataloader, device, num_samples=100):
    model.eval()
    rouge = load_metric('rouge')
    references = []
    predictions = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            
            # Generate text using greedy decoding
            outputs = model.generate(input_ids, max_length=50)
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            predictions.extend(generated_texts)
            references.extend(prompts)
    
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    return rouge_scores
