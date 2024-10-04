# tokenizers/tokenizer.py

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk

def train_tokenizer(processed_data_path, tokenizer_save_path='./tokenizer/', vocab_size=30000, min_frequency=2):
    # Load processed dataset
    dataset = load_from_disk(processed_data_path)
    train_texts = dataset['train']['text']
    
    # Initialize the tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Train the tokenizer
    tokenizer.train_from_iterator(
        train_texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
            "<MASK>",
        ]
    )
    
    # Save the tokenizer
    tokenizer.save_model(tokenizer_save_path)
    
    # Convert to PreTrainedTokenizerFast
    tokenizer_fast = PreTrainedTokenizerFast(tokenizer_file=f"{tokenizer_save_path}/tokenizer.json",
                                            unk_token="<UNK>",
                                            pad_token="<PAD>",
                                            bos_token="<BOS>",
                                            eos_token="<EOS>",
                                            mask_token="<MASK>")
    tokenizer_fast.save_pretrained(tokenizer_save_path)
    print(f"Tokenizer trained and saved to {tokenizer_save_path}")

def load_trained_tokenizer(tokenizer_path='./tokenizer/'):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer
