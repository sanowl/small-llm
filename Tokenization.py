from tokenizers import ByteLevelBPETokenizer
import dataset

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer on the dataset
def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]['text']

# Train the tokenizer
tokenizer.train_from_iterator(
    batch_iterator(dataset['train']),
    vocab_size=30_000,
    min_frequency=2,
    special_tokens=[
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
        "<MASK>",
    ]
)

# Save the tokenizer
tokenizer.save_model('./tokenizer/')
