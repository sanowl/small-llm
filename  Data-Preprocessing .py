import re

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters except for standard punctuation
    text = re.sub(r'[^A-Za-z0-9\s\.,;:\'\"\!\?()-]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning to the dataset
dataset = dataset.map(lambda x: {'text': clean_text(x['text'])}, batched=True)
