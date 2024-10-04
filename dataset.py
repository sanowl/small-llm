from datasets import load_dataset

# Load the 'scientific_papers' dataset
dataset = load_dataset('scientific_papers')

# Inspect the dataset splits
print(dataset)

# Example Output:
# DatasetDict({
#     train: Dataset({
#         features: ['abstract', 'body_text', 'title', 'url'],
#         num_rows: 100000
#     })
#     validation: Dataset({
#         features: ['abstract', 'body_text', 'title', 'url'],
#         num_rows: 20000
#     })
#     test: Dataset({
#         features: ['abstract', 'body_text', 'title', 'url'],
#         num_rows: 20000
#     })
# })
